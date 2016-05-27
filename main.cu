/**
 * Computación Paralela (curso 1516)
 *
 * Colocación de antenas
 * Versión paralela
 *
 * @author Lucía Gil
 * @author Jorge Hernández
 */


// Includes generales
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <cuda.h>
#include <cutil.h>
// Include para las utilidades de computación paralela
#include "cputils.h"

// Definición de constantes
#define currentGPU 0
#define BLOCK_DIM_FILAS 128
#define BLOCK_DIM_COLUMNAS 8
#define MAX_THREADS 1024

#define m(y,x) mapa[ (y * cols) + x ]
#define d_mapa(y,x) mapa[ (y * numcols) + x ]

// Vectores para realizar las reducciones y sus funciones de reserva y liberación de memoria.
static int* row_Result;
static int* col_Result;
static int* val_Result;

void reduceAllocate(const unsigned int num_blocks){
    CUDA_SAFE_CALL(cudaMalloc((void**)&val_Result, BLOCK_DIM_FILAS * num_blocks * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**)&row_Result, num_blocks * sizeof(int))); 
    CUDA_SAFE_CALL(cudaMalloc((void**)&col_Result, BLOCK_DIM_FILAS * num_blocks * sizeof(int)));
}

void reduceFree(){
    CUDA_SAFE_CALL(cudaFree(val_Result));
    CUDA_SAFE_CALL(cudaFree(row_Result));
    CUDA_SAFE_CALL(cudaFree(col_Result));
}

/**
 * Estructura antena
 */
typedef struct {
	int fila;
	int columna;
} Antena;

// Declaración de prototipos de kernels y funciones internas:
__global__ void update(int*, const unsigned int, const unsigned int, const unsigned int, const unsigned int);

__global__ void reduce_columns_kernel(const int*, const unsigned int, const unsigned int, int*,int*);

__global__ void reduce_rows_kernel( int*, int*, int*);

__global__ void reduce_blocks_kernel(int* vals, int* rows, int* cols, const unsigned int lastBlock, const int reductionLevel, const int nuevas);

__device__ int distanciaManhattan( int antena_fila, int antena_columna, int fila, int columna) {
    int dist = abs(antena_fila - fila) + abs(antena_columna - columna);
    return dist * dist;
}
//
int* maxDistance(const int* mapa, const unsigned int numfil, const unsigned int numcols, const dim3 grid, const dim3 block, const int nuevas);

//
void print_mapa(int * mapa, int rows, int cols){


	/*if(rows > 50 || cols > 30){
		printf("Mapa muy grande para imprimir\n");
		return;
	};*/

	#define ANSI_COLOR_RED     "\x1b[31m"
	#define ANSI_COLOR_GREEN   "\x1b[32m"
	#define ANSI_COLOR_RESET   "\x1b[0m"

	printf("Mapa [%d,%d]\n",rows,cols);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){

			int val = m(i,j);
			printf("%4d",val);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * Función principal
 */
int main(int nargs, char ** vargs){

	cudaSetDevice(currentGPU);
	// Comprobar número de argumentos
	if(nargs < 7){
		fprintf(stderr,"Uso: %s rows cols distMax nAntenas x0 y0 [x1 y1, ...]\n",vargs[0]);
		return -1;
	}

	//
	// 1. LEER DATOS DE ENTRADA
	//

	// Leer los argumentos de entrada
	int rows = atoi(vargs[1]);
	int cols = atoi(vargs[2]);
	int distMax = atoi(vargs[3]);
	int nAntenas = atoi(vargs[4]);

	//printf("nAntenas: %d\n", nAntenas);
	if(nAntenas<1 || nargs != (nAntenas*2+5)){
		fprintf(stderr,"Error en la lista de antenas\n");
		return -1;
	}
	// Reservar memoria para las antenas
	Antena * antenas = (Antena *) malloc(sizeof(Antena) * (size_t) nAntenas);
	if(!antenas){
		fprintf(stderr,"Error al reservar memoria para las antenas inicales\n");
		return -1;
	}	
	// Leer antenas
	for(int i=0; i<nAntenas; i++){
		antenas[i].columna = atoi(vargs[5+i*2]);
		antenas[i].fila = atoi(vargs[6+i*2]);

		if(antenas[i].fila<0 || antenas[i].fila>=rows || antenas[i].columna<0 || antenas[i].columna>=cols ){
			fprintf(stderr,"Antena #%d está fuera del mapa\n",i);
			return -1;
		}
	}

	/*printf("Calculando el número de antenas necesarias para cubrir un mapa de"
		   " (%d x %d)\ncon una distancia máxima no superior a %d "
		   "y con %d antenas iniciales\n\n",rows,cols,distMax,nAntenas);*/

	//
	// 2. INICIACIÓN
	//

	// Medir el tiempo
	double tiempo = cp_Wtime();
	
	int * h_mapa;
	int * d_mapa;

	// Allocate host memory for the values
	h_mapa = (int*) malloc((size_t) (rows * cols) * sizeof(int) );

	// Iniciar el mapa con el valor MAX INT
	for(int i=0; i<(rows*cols); i++){
		h_mapa[i] = INT_MAX;
	}

	cudaError_t error;

	// Allocate device memory for the values
	cudaMalloc( (void**) &d_mapa, sizeof(int) * (int) (rows*cols));
	
	// Copy values from host memory to device memory
	error = cudaMemcpy(d_mapa, h_mapa, sizeof(int)*rows*cols, cudaMemcpyHostToDevice);
	if(error != cudaSuccess)
		printf("Error al copiar del host al device: %s\n", cudaGetErrorString( error ) );
	//else
	//	printf("Datos escritos en host\n");
	
	// Definicion de los tamaños de bloque y grid:
	dim3 block(BLOCK_DIM_COLUMNAS,BLOCK_DIM_FILAS);
	int num_col_grid, num_fil_grid;
	if( cols % BLOCK_DIM_COLUMNAS != 0)
		num_col_grid = cols/BLOCK_DIM_COLUMNAS+1;
	else
		num_col_grid = cols/BLOCK_DIM_COLUMNAS;
	if( rows % BLOCK_DIM_FILAS != 0)
		num_fil_grid = rows/BLOCK_DIM_FILAS+1;
	else
		num_fil_grid = rows/BLOCK_DIM_FILAS;

	//printf("F%d-C%d\n", num_fil_grid, num_col_grid);
	dim3 grid(num_col_grid, num_fil_grid);

	int num_blocks = num_col_grid * num_fil_grid;
	// Colocar las antenas iniciales
	for(int i=0; i<nAntenas; i++){
		//printf("Colocando una antena en %d %d\n",antenas[i].fila, antenas[i].columna);
		update<<< grid, block>>>(d_mapa, antenas[i].fila, antenas[i].columna, rows, cols);
		if((error=cudaGetLastError()) != cudaSuccess)
			printf("Error al colocar las primeras antenas: %s\n", cudaGetErrorString( error ) );
	}

	//
	// 3. CALCULO DE LAS NUEVAS ANTENAS
	//

	// Contador de antenas
	int nuevas = 0;
	int *antenaHost = (int*) malloc((size_t) 3 * sizeof(int) );
	reduceAllocate(num_blocks);
	while(1){
        //antenaHost = maxDistance(d_mapa,rows, cols, grid, block);
		antenaHost = maxDistance(d_mapa,rows, cols, grid, block, nuevas);
		// Traemos la nueva antena: [ Distancia, Fila y Columna ]
		int dist = antenaHost[0];
		int fil = antenaHost[1];
		int col = antenaHost[2];
		//printf("[%d-%d,%d]\t",dist,fil,col);
		if(dist <= distMax){
			break;
		}
		nuevas++;
		update<<< grid, block>>>(d_mapa, fil, col, rows, cols);
	}

	//
	// 4. MOSTRAR RESULTADOS
	//

	// tiempo
	tiempo = cp_Wtime() - tiempo;	

	// Salida
	printf("Result: %d\n",nuevas);
	printf("Time: %f\n",tiempo);
	
	reduceFree();
	cudaDeviceReset();
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
//! Kernel que recibe un mapa de distancias, coloca una antena en una posición y actualiza el mapa original.
//! @param mapa         Matriz con distancias a antenas
//! @param fila         Num de fila donde se ha colocado una antena
//! @param columna      Num de columna donde se ha colocado una antena
//! @param numcols      Num de elementos por columna del mapa original
//! @param numfil       Num de filas del mapa original
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void update( int* mapa, const unsigned int a_fila, const unsigned int a_columna,
                         const unsigned int numfil, const unsigned int numcols){
    // Obtenemos el elemento del mapa al que realmente ha de acceder este hilo
    unsigned int tid_row = threadIdx.y;
    unsigned int tid_col = threadIdx.x;
    
    unsigned int real_row = blockIdx.y * blockDim.y + tid_row;
    unsigned int real_col = blockIdx.x * blockDim.x + tid_col;
    
    // Algunos hilos podrían estar accediendo a elementos que no existen. Impedimos esto:
    if(real_col < numcols && real_row < numfil){
        int dist = distanciaManhattan(a_fila,a_columna, real_row, real_col);
        //printf("Thread [%d, %d] - Bloque [%d, %d] - Elemento[%d,%d]\nDistance to antenna: %d\n",tid_row,tid_col, blockIdx.y, blockIdx.x, real_fila, real_col, dist);
        if(d_mapa(real_row,real_col) > dist)
            d_mapa(real_row, real_col) = dist;
        
    }
}


__global__ void reduce_columns_kernel(const int* mapa, const unsigned int numfil, const unsigned int numcols, int* values, int* cols){
    extern __shared__ unsigned int sdata[];
    // Obtenemos el elemento del mapa al que realmente ha de acceder este hilo
    unsigned int tid_row = threadIdx.y;
    unsigned int tid_col = threadIdx.x;

    unsigned int real_row = blockIdx.y * blockDim.y + tid_row;
    unsigned int real_col = blockIdx.x * blockDim.x + tid_col;
    
    // Obtenemos un identificador único para el hilo dentro de su bloque
    unsigned int tid = blockDim.x * tid_row + tid_col;

    // Inicialización de la reducción
    if(real_col < numcols && real_row < numfil)
    	sdata[2*tid] = d_mapa(real_row,real_col);
	else
		sdata[2*tid] = 0;		
	sdata[2*tid+1] = real_col;
    __syncthreads();
    
    //Hacemos la reducción en memoria shared por filas hasta obtener una columna única 
    for (unsigned int s=blockDim.x/2; s>0; s/=2) {
        if (tid_col < s) {
        	// Hacemos la reducción sumando los dos elementos que le tocan a este hilo
        	if(sdata[2*tid] < sdata[2*(tid+s)]){
        		sdata[2*tid] = sdata[2*(tid+s)];
        		sdata[2*tid+1] = sdata[2*(tid+s)+1];
        	}
        	if(sdata[2*tid] == sdata[2*(tid+s)]){
        	   if(sdata[2*tid+1] > sdata[2*(tid+s)+1])
        	       sdata[2*tid+1] = sdata[2*(tid+s)+1];
        	}
    	}
        __syncthreads();
    }
    
    //Tenemos los maximos en la primera columna
    if (tid_col == 0){
    	//Num de bloque?
    	unsigned int offset = blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x);
        //printf("Voy a guardar mi maximo en la pos: %d\n",offset+tid_row);
        values[offset + tid_row] =  sdata[2*tid];
        cols[offset + tid_row] = sdata[2*tid+1];
    }
}

__global__ void reduce_rows_kernel( int* vals, int* rows, int *cols ){
    extern __shared__ unsigned int sdata[];
    // Identificador de hilo
    unsigned int tid = threadIdx.x;
    
    // Localizado a la altura de la matriz:
    unsigned int real_row = blockDim.x * blockIdx.y + tid;
    
    // Hemos guardado los máximos locales desplazados:
    unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int offset = blockDim.x * blockId;

    sdata[3*tid]   = vals[ offset + tid ]; // si te has pasao de fila habra un 0, its safe
    sdata[3*tid+1] = real_row;
    sdata[3*tid+2] = cols[ offset + tid ];

    //Sincronizamos para que todos hayan llegado a este punto
    __syncthreads();
    
    // Hacemos la reducción en memoria shared
    for (unsigned int s=blockDim.x/2; s>0; s/=2) {
        if (tid < s) {
            if(sdata[3*tid] < sdata[3*(tid+s)]){
                sdata[3*tid] = sdata[3*(tid+s)];
                sdata[3*tid+1] = sdata[3*(tid+s)+1];
                sdata[3*tid+2] = sdata[3*(tid+s)+2];
            }
            if(sdata[3*tid] == sdata[3*(tid+s)]){
                //Actualizo en el caso de que su fila sea menor:
                if(sdata[3*tid+1] > sdata[3*(tid+s)+1]){
                    sdata[3*tid+1] = sdata[3*(tid+s)+1];
                    sdata[3*tid+2] = sdata[3*(tid+s)+2];
                }
            }
        }
        __syncthreads();
    }

    if(tid == 0){
        vals[offset]  = sdata[0];
        rows[blockId] = sdata[1];//identificador del bloque dentro del grid;
        cols[offset]  = sdata[2];
    }
}

__global__ void reduce_blocks_kernel(int* vals, int* rows, int* cols, const unsigned int lastBlock, const int reductionLevel, const int nuevas){
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    // Si estamos en el bloque 0, nos corresponden los bloques originales 0-1023.
    // En el bloque 1 nos corresponden del 1024-2047, etc.
    unsigned int offset; 
    unsigned int bid;
    if(reductionLevel == 1){
        bid = tid;
        offset = blockIdx.x * MAX_THREADS;
        bid += offset;
    }
    if(reductionLevel == 2){ // Suponiendo que solo hay un bloque :S
        bid = tid * MAX_THREADS ;
        offset = tid * MAX_THREADS;
    }
    
    //if(reductionLevel == 1 && nuevas == 0 && tid == 0)
    //    printf("Hilo: %d Bloque: %d Posiciones: %d y %d\n",tid, bid, BLOCK_DIM_FILAS * offset, offset);
    //if(reductionLevel == 2 && nuevas == 0)
    //    printf("Hilo: %d Bloque: %d Posiciones: %d y %d\n",tid, bid, BLOCK_DIM_FILAS * offset, offset);

    sdata[3*tid]   = 0;
    sdata[3*tid+1] = 0;
    sdata[3*tid+2] = 0;
    
    if(bid < lastBlock && reductionLevel == 1){
        sdata[3*tid]   = vals[ BLOCK_DIM_FILAS * bid ];
        sdata[3*tid+1] = rows[ bid ];
        sdata[3*tid+2] = cols[ BLOCK_DIM_FILAS * bid ];
    }
    
    if(tid < lastBlock && reductionLevel == 2){
        sdata[3*tid]   = vals[ BLOCK_DIM_FILAS * offset ];
        sdata[3*tid+1] = rows[ offset ];
        sdata[3*tid+2] = cols[ BLOCK_DIM_FILAS * offset ];
    }
    __syncthreads();
    if(reductionLevel == 1){
        for (unsigned int s=MAX_THREADS/2; s>0; s/=2) {
            if (tid < s && bid < lastBlock && (bid+s) < lastBlock) {
                if(sdata[3*tid] < sdata[3*(tid+s)]){
                    sdata[3*tid] = sdata[3*(tid+s)];
                    sdata[3*tid+1] = sdata[3*(tid+s)+1];
                    sdata[3*tid+2] = sdata[3*(tid+s)+2];
                }
                else if(sdata[3*tid] == sdata[3*(tid+s)]){
                    //Actualizo en el caso de que su fila sea menor:
                    
                    if(sdata[3*tid+1] > sdata[3*(tid+s)+1]){
                        sdata[3*tid+1] = sdata[3*(tid+s)+1];
                        sdata[3*tid+2] = sdata[3*(tid+s)+2];
                    }
                    //Si son de la misma fila, actualizo si la columna es mayor
                    else if(sdata[3*tid+1] == sdata[3*(tid+s)+1] && sdata[3*tid+2] > sdata[3*(tid+s)+2]){
                        sdata[3*tid+2] = sdata[3*(tid+s)+2];
                    }                
                }
            }
            __syncthreads();
        }
    }
    else{
        for (unsigned int s=MAX_THREADS/2; s>0; s/=2) {
            if (tid < s && tid < lastBlock && (tid+s) < lastBlock) {
                if(sdata[3*tid] < sdata[3*(tid+s)]){
                    sdata[3*tid] = sdata[3*(tid+s)];
                    sdata[3*tid+1] = sdata[3*(tid+s)+1];
                    sdata[3*tid+2] = sdata[3*(tid+s)+2];
                }
                else if(sdata[3*tid] == sdata[3*(tid+s)]){
                    //Actualizo en el caso de que su fila sea menor:
                    
                    if(sdata[3*tid+1] > sdata[3*(tid+s)+1]){
                        sdata[3*tid+1] = sdata[3*(tid+s)+1];
                        sdata[3*tid+2] = sdata[3*(tid+s)+2];
                    }
                    //Si son de la misma fila, actualizo si la columna es mayor
                    else if(sdata[3*tid+1] == sdata[3*(tid+s)+1] && sdata[3*tid+2] > sdata[3*(tid+s)+2]){
                        sdata[3*tid+2] = sdata[3*(tid+s)+2];
                    }                
                }
            }
            __syncthreads();
        }    
    }

    if(tid == 0){
        vals[ BLOCK_DIM_FILAS * offset ] = sdata[0];
        rows[ offset ] = sdata[1];
        cols[ BLOCK_DIM_FILAS * offset ] = sdata[2];
        /*if(reductionLevel == 1 && blockIdx.x == 3 && nuevas == 0)
            printf("Bloque: %d Posiciones: %d y %d\n", bid,BLOCK_DIM_FILAS * offset, offset);
        if(reductionLevel == 2 && nuevas == 0)
            printf("Bloque: %d Posiciones: %d y %d\n", bid,BLOCK_DIM_FILAS * offset, offset);*/
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
//! Función que se encarga de lanzar los kernels para buscar donde colocar la nueva antena
//! @param mapa         Matriz con distancias a antenas
//! @param numcols      Num de elementos por columna del mapa original
//! @param numfil       Num de filas del mapa original
//! @param num_blocks   Num de bloques sobre los que se hace la reduccion.
//! @return int*        Vector de tres elementos: Distancia, Fila y Columna
///////////////////////////////////////////////////////////////////////////////////////////
//int* maxDistance(const int* mapa, const unsigned int numfil, const unsigned int numcols, const dim3 grid, const dim3 block, const int nuevas){
int* maxDistance(const int* mapa, const unsigned int numfil, const unsigned int numcols, const dim3 grid, const dim3 block, const int nuevas){
    cudaError_t error;
    int numThreadsPerBlock = block.y * block.x; // NUMERO DE FILAS * NUMERO DE COLUMNAS

    // En la primera pasada cada bloque de 32 x 32 vamos a reducir por filas el maximo.
    // Por tanto para hacer la reducción es necesario que cada hilo tenga el valor maximo y la columna
    // donde se encontro dicho máximo, ya que la fila es constante.
    int sharedMemorySize = 2 * numThreadsPerBlock * sizeof(int);

    reduce_columns_kernel<<<grid, block, sharedMemorySize>>>(mapa, numfil, numcols,val_Result, col_Result);
    if((error=cudaGetLastError()) != cudaSuccess)
        printf("Error al llamar al primer reduce kernel: %s\n", cudaGetErrorString( error ) );
    CUT_CHECK_ERROR("Kernel execution failed");

    // En la segunda pasada cada bloque accederá a los 32 máximos locales encontrados previamente, y hará reducción
    // hasta obtener en la primera posición el máximo de su bloque. Para ello es necesario que los hilos de las reducciones
    // sepan el valor de la distancia, de la fila y de la columna mejores encontrados.
    numThreadsPerBlock = block.y; // ES EL NUMERO DE FILAS
    sharedMemorySize = 3 * numThreadsPerBlock * sizeof(int);
    reduce_rows_kernel<<< grid, numThreadsPerBlock, sharedMemorySize>>>(val_Result, row_Result, col_Result);
    if((error=cudaGetLastError()) != cudaSuccess)
        printf("Error al llamar al segundo reduce kernel: %s\n", cudaGetErrorString( error ) );
    CUT_CHECK_ERROR("Kernel execution failed");
    
    // En la tercera pasada asignamos un bloque con hasta 1024 hilos para cada conjunto de hasta 1024 bloques con sus
    // máximos. Ésta puede ser la última reducción, en el caso de que haya menos de 1024 bloques, pero si hay mas será necesario
    // hacer más reducciones.
    int inception = 0;
    int blocks;
    numThreadsPerBlock = grid.x * grid.y;
    int lastBlock = grid.x * grid.y;
    do{
        inception++;
        if(numThreadsPerBlock > 1024){
            if(!(numThreadsPerBlock%1024) )
                blocks = numThreadsPerBlock/1024;
            else
                blocks = numThreadsPerBlock/1024 + 1;

            sharedMemorySize = 3 * MAX_THREADS * sizeof(int);
            //reduce_blocks_kernel<<<blocks, MAX_THREADS, sharedMemorySize>>>(val_Result, row_Result, col_Result, lastBlock, inception, nuevas);
            reduce_blocks_kernel<<<blocks, MAX_THREADS, sharedMemorySize>>>(val_Result, row_Result, col_Result, blocks * 1024 + numThreadsPerBlock % 1024, inception, nuevas);
            //if(!nuevas) printf("%d-%d-%d\n", inception, blocks, MAX_THREADS);
            numThreadsPerBlock = blocks;
        }
        else{//Última reducción
            blocks = 1;
            sharedMemorySize = 3 * numThreadsPerBlock * sizeof(int);
            reduce_blocks_kernel<<<blocks, numThreadsPerBlock, sharedMemorySize>>>(val_Result, row_Result, col_Result, numThreadsPerBlock, inception, nuevas);
            //reduce_blocks_kernel<<<blocks, numThreadsPerBlock, sharedMemorySize>>>(val_Result, row_Result, col_Result, last, inception, nuevas);
            //if(!nuevas) printf("%d-%d-%d\n", inception, blocks, numThreadsPerBlock);
        }
        
    }
    while(blocks > 1);
    int *result = (int*) malloc((size_t) 3 * sizeof(int) );
    
    cudaMemcpy(result, val_Result,  sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(result[1]), row_Result,  sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(result[2]), col_Result,  sizeof(int), cudaMemcpyDeviceToHost);

    return result;
}
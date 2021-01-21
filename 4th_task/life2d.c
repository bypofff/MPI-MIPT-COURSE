/*
 * Author: Nikolay Khokhlov <k_h@inbox.ru>, 2016
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))

typedef struct {
	int nx, ny;
	int *u0;
	int *u1;
	int steps;
	int save_steps;
	
	/* MPI */
	int begin_k, end_k; /* Начало и конец зоны ответсвенности текущего (k-го) процесса. [begin_k; end_k) */
	int rank; /* Номер текущего процесса. */
	int size; /* Число процессов. */

	MPI_Datatype block_type; /*тип данных зоны ответственности кроме последнего*/

	MPI_Datatype vertical_type; /*Тип данных в столбце поля*/
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);

/*
 * Декомпозиция по одной оси.
 * p - число процессов;
 * k - номер текущего процесса;
 * n - размер области.
 */
void life_decomposition(const int p, const int k, const int n, int *begin, int *end);

/*
 * Сбор у последнего процесса.
 */
void life_collect(life_t *l);

/*
 * Обмен между процессами.
 */

void exchange(life_t *l);

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}
	MPI_Init(&argc, &argv);
	life_t l;
	life_init(argv[1], &l);
	
	int i;
	char buf[100];

	FILE *fout;
    	fout = fopen("results.txt", "a");
	double start_time =  MPI_Wtime();

	for (i = 0; i < l.steps; i++) {
		if (i % l.save_steps == 0) {
			life_collect(&l);
			if (l.rank == l.size - 1) {
				sprintf(buf, "life_%06d.vtk", i);
				printf("Saving step %d to '%s'.\n", i, buf);
				life_save_vtk(buf, &l);
			}
		}
		life_step(&l);
		exchange(&l);
	}
	double finish_time = MPI_Wtime() - start_time;
	
	if (l.rank == 0){
		fprintf(fout,"%e\n", finish_time);
	}

	life_free(&l);
	MPI_Finalize();
	return 0;
}

/**
 * Загрузить входную конфигурацию.
 * Формат файла, число шагов, как часто сохранять, размер поля, затем идут координаты заполненых клеток:
 * steps
 * save_steps
 * nx ny
 * i1 j2
 * i2 j2
 */
void life_init(const char *path, life_t *l)
{
	FILE *fd = fopen(path, "r");
	assert(fd);
	assert(fscanf(fd, "%d\n", &l->steps));
	assert(fscanf(fd, "%d\n", &l->save_steps));
	//printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));
	//printf("Field size: %dx%d\n", l->nx, l->ny);

	l->u0 = (int*)calloc(l->nx * l->ny, sizeof(int));
	l->u1 = (int*)calloc(l->nx * l->ny, sizeof(int));
	
	int i, j, r, cnt;
	cnt = 0;
	while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
		l->u0[ind(i, j)] = 1;
		cnt++;
	}
	fclose(fd);
	
	MPI_Comm_size(MPI_COMM_WORLD, &(l->size));
	MPI_Comm_rank(MPI_COMM_WORLD, &(l->rank));
	life_decomposition(l->size, l->rank, l->nx, &(l->begin_k), &(l->end_k));

	int begin, end;
	life_decomposition(l->size, 0, l->nx, &begin, &end);
	MPI_Type_vector(l->ny, end-begin, l->nx, MPI_INT, &(l->block_type));
	MPI_Type_commit(&(l->block_type));

	MPI_Type_vector(l->ny, 1, l->nx, MPI_INT, &(l->vertical_type));
	MPI_Type_commit(&(l->vertical_type));
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	l->nx = l->ny = 0;
	MPI_Type_free(&(l->block_type));
	MPI_Type_free(&(l->vertical_type));
}

void life_save_vtk(const char *path, life_t *l)
{
	FILE *f;
	int i1, i2, j;
	f = fopen(path, "w");
	assert(f);
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by write_to_vtk2d\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d %d 1\n", l->nx+1, l->ny+1);
	fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
	fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
	fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);
	
	fprintf(f, "SCALARS life int 1\n");
	fprintf(f, "LOOKUP_TABLE life_table\n");
	for (i2 = 0; i2 < l->ny; i2++) {
		for (i1 = 0; i1 < l->nx; i1++) {
			fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
		}
	}
	fclose(f);
}

void life_step(life_t *l)
{
	int i, j;
	for (j = 0; j < l->ny; j++) {
		for (i = l->begin_k; i < l->end_k; i++) {
			int n = 0;
			n += l->u0[ind(i+1, j)];
			n += l->u0[ind(i+1, j+1)];
			n += l->u0[ind(i,   j+1)];
			n += l->u0[ind(i-1, j)];
			n += l->u0[ind(i-1, j-1)];
			n += l->u0[ind(i,   j-1)];
			n += l->u0[ind(i-1, j+1)];
			n += l->u0[ind(i+1, j-1)];
			l->u1[ind(i,j)] = 0;
			if (n == 3 && l->u0[ind(i,j)] == 0) {
				l->u1[ind(i,j)] = 1;
			}
			if ((n == 3 || n == 2) && l->u0[ind(i,j)] == 1) {
				l->u1[ind(i,j)] = 1;
			}
			/* Проверка сбора данных. */
			// l->u1[ind(i,j)] = l->rank;
		}
	}
	int *tmp;
	tmp = l->u0;
	l->u0 = l->u1;
	l->u1 = tmp;
}

void life_decomposition(const int p, const int k, const int n, int *begin, int *end)
{
	*begin = k * (n / p);
	*end = *begin + (n / p);
	if (k == p - 1) *end = n;
}

void life_collect(life_t *l)
{
	if (l->rank == l->size - 1) {
		int i;
		for (i = 0; i < l->size - 1; i++) {
			int begin_i, end_i;
			life_decomposition(l->size, i, l->nx, &begin_i, &end_i);
			MPI_Recv(l->u0 + ind(begin_i, 0), 1, l->block_type, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	} else {
		MPI_Send(l->u0 + ind(l->begin_k, 0), 1, l->block_type, l->size - 1, 0, MPI_COMM_WORLD);
	}
}

void exchange(life_t *l)
{
	if (l->size != 1) {
		if (l->rank % 2 == 0) {
			MPI_Send(l->u0 + ind(l->end_k - 1, 0), 1, l->vertical_type, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD);
        	MPI_Recv(l->u0 + ind(l->begin_k - 1, 0), 1, l->vertical_type, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	MPI_Send(l->u0 + ind(l->begin_k, 0), 1, l->vertical_type, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD);
    		MPI_Recv(l->u0 + ind(l->end_k, 0), 1, l->vertical_type, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(l->u0 + ind(l->begin_k - 1, 0), 1, l->vertical_type, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            		MPI_Send(l->u0 + ind(l->end_k - 1, 0), 1, l->vertical_type, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD);
            		MPI_Recv(l->u0 + ind(l->end_k, 0), 1, l->vertical_type, (l->rank + 1) % l->size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            		MPI_Send(l->u0 + ind(l->begin_k, 0), 1, l->vertical_type, (l->rank - 1 + l->size) % l->size, 0, MPI_COMM_WORLD);
		}
	}
}


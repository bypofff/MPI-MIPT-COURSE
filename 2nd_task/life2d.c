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
	
	int rank;
	int size;
	int begin_y;
	int end_y;
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void life_decomposition(const int n, const int rank, const int size, int *begin, int *end);
void life_gather(life_t *l);
void life_exchange(life_t *l);

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}
	MPI_Init(&argc, &argv);
	life_t l;
	life_init(argv[1], &l);
	
	double tm = MPI_Wtime();
	int i;
	char buf[100];
	for (i = 0; i < l.steps; i++) {
		
		if (i % l.save_steps == 0) {
			life_gather(&l);
			if (l.rank == 0) {
				sprintf(buf, "vtk/life_%06d.vtk", i);
				printf("Saving step %d to '%s'.\n", i, buf);
				life_save_vtk(buf, &l);
			}
			
		}
		
		life_step(&l);
		life_exchange(&l);
	}
	
	if(l.rank == 0) {
		printf("%lf\n", MPI_Wtime() - tm);
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
	//printf("Loaded %d life cells.\n", cnt);
	fclose(fd);
	
	
	MPI_Comm_rank(MPI_COMM_WORLD, &l->rank);
	MPI_Comm_size(MPI_COMM_WORLD, &l->size);
	life_decomposition(l->ny, l->rank, l->size, &l->begin_y, &l->end_y);
	//printf("#%d: begin_y = %d, end_y = %d\n", l->rank, l->begin_y, l->end_y);
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	l->nx = l->ny = 0;
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
	#pragma omp parallel
	{
	int i, j;
	for (j = l->begin_y; j < l->end_y; j++) {
		for (i = 0; i < l->nx; i++) {
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
			
			//l->u1[ind(i,j)] = l->rank;
		}
	}
	}
	int *tmp;
	tmp = l->u0;
	l->u0 = l->u1;
	l->u1 = tmp;
}

void life_decomposition(const int n, const int rank, const int size, int *begin, int *end)
{
	*begin = (n / size) * rank;
	*end = *begin + (n / size);
	if (rank == size - 1) *end = n;
}

void life_gather(life_t *l)
{
	if (l->rank == 0) {
		int i;
		int begin_i = 0, end_i = 0;
		for (i = 1; i < l->size; i++) {
			life_decomposition(l->ny, i, l->size, &begin_i, &end_i);
			MPI_Recv(&(l->u0[ind(0, begin_i)]), 
				l->nx * (end_i - begin_i), MPI_INT, 
				i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	} else {
		MPI_Send(&(l->u0[ind(0, l->begin_y)]), 
				l->nx * (l->end_y - l->begin_y), MPI_INT, 
				 0, 0, MPI_COMM_WORLD);
	}
}

void life_exchange(life_t *l) {
    if (l->size > 1) {
		int prev_send = (l->rank == 0 ? l->size - 1 : l->rank - 1);
		int right_send = (l->rank + 1) % l->size; 
		int left_receive = (l->rank == 0 ? l->ny - 1 : l->begin_y - 1);
		int right_receive = (l->rank + 1 == l->size ? 0 : l->end_y);
		
		if (l->rank % 2 == 0) {
			MPI_Send(&(l->u0[ind(0, l->end_y - 1)]), l->nx, MPI_INT, right_send, 0, MPI_COMM_WORLD); 
			MPI_Recv(&(l->u0[ind(0, right_receive)]), l->nx, MPI_INT, right_send, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
			MPI_Recv(&(l->u0[ind(0, left_receive)]), l->nx, MPI_INT, prev_send, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&(l->u0[ind(0, l->begin_y)]), l->nx, MPI_INT, prev_send, 0, MPI_COMM_WORLD);
		} else {
			MPI_Recv(&(l->u0[ind(0, left_receive)]), l->nx, MPI_INT, prev_send, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
			MPI_Send(&(l->u0[ind(0, l->begin_y)]), l->nx, MPI_INT, prev_send, 0, MPI_COMM_WORLD);
			MPI_Send(&(l->u0[ind(0, l->end_y - 1)]), l->nx, MPI_INT, right_send, 0, MPI_COMM_WORLD); 
			MPI_Recv(&(l->u0[ind(0, right_receive)]), l->nx, MPI_INT, right_send, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
		}
	}
}
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
    int begin_k[2], end_k[2]; /* Начало и конец зоны ответсвенности текущего (k-го) процесса. [begin_k; end_k) */
    int rank; /* Номер текущего процесса. */
    int size; /* Число процессов. */
    int dims[2]; /* Размерность декартовой топологии в процессах. */
    MPI_Comm cart_comm; /* Коммуникатор, описывающий декартову топологию. */
    MPI_Datatype block_type; /* Тип данных зоны ответственности всех процессов, кроме последнего. */
    MPI_Datatype line_type; /* Тип данных зоны ответственности всех процессов, кроме последнего. */
    MPI_Datatype column_type; /* Тип данных зоны ответственности всех процессов, кроме последнего. */
    int coords[2]; /* Координаты процесса в новой топологии. */
    MPI_Comm dims_comm[2]; /* Строчные и столбцовые коммуникаторы. */
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

void life_exchange(life_t *l);

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s input file.\n", argv[0]);
        return 0;
    }
    MPI_Init(&argc, &argv);
    life_t l;
    life_init(argv[1], &l);

    int i;
    char buf[100];
    for (i = 0; i < l.steps; ++i) {
        if (i % l.save_steps == 0) {
            life_collect(&l);
            if (l.rank == l.size - 1) {
                sprintf(buf, "life_%06d.vtk", i);
                printf("Saving step %d to '%s'.\n", i, buf);
                life_save_vtk(buf, &l);
            }
        }
        life_exchange(&l);
        life_step(&l);
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
void life_init(const char *path, life_t *l) {
    FILE *fd = fopen(path, "r");
    assert(fd);
    assert(fscanf(fd, "%d\n", &l->steps));
    assert(fscanf(fd, "%d\n", &l->save_steps));
    //printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
    assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));
    //printf("Field size: %dx%d\n", l->nx, l->ny);

    l->u0 = (int *) calloc(l->nx * l->ny, sizeof(int));
    l->u1 = (int *) calloc(l->nx * l->ny, sizeof(int));

    int i, j, r, cnt;
    cnt = 0;
    while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
        l->u0[ind(i, j)] = 1;
        ++cnt;
    }
    //printf("Loaded %d life cells.\n", cnt);
    fclose(fd);

    /* Декомпозиция. */
    MPI_Comm_size(MPI_COMM_WORLD, &(l->size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(l->rank));
    l->dims[0] = l->dims[1] = 0;
    MPI_Dims_create(l->size, 2, l->dims);
    int reorder = 0;
    int periods[2] = {1, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, l->dims, periods, reorder, &(l->cart_comm));
    MPI_Cart_coords(l->cart_comm, l->rank, 2, l->coords);
    printf("#%d: %d, %d\n", l->rank, l->coords[0], l->coords[1]);

    int n[2] = {l->nx, l->ny};
    for (int k = 0; k < 2; ++k) {
        life_decomposition(l->dims[k], l->coords[k], n[k], &(l->begin_k[k]), &(l->end_k[k]));
    }

    MPI_Comm_split(l->cart_comm, l->coords[0], l->coords[1], &(l->dims_comm[0]));
    MPI_Comm_split(l->cart_comm, l->coords[1], l->coords[0], &(l->dims_comm[1]));

    /* Создаем тип блока. */
    int begin[2], end[2];
    for (int k = 0; k < 2; ++k) {
        life_decomposition(l->dims[k], l->coords[k] * (k == 1 ? 1 : 0), n[k], begin + k, end + k);
    }

    MPI_Type_vector(end[1] - begin[1], end[0] - begin[0], l->nx, MPI_INT, &(l->block_type));
    MPI_Type_commit(&(l->block_type));

    MPI_Type_contiguous((end[1] - begin[1]) * l->nx, MPI_INT, &(l->line_type));
    MPI_Type_commit(&(l->line_type));

    // МЕМ
    // MPI_Type_vector(end[0] - begin[0], 1, l->ny, MPI_INT, &(l->column_type));
    // MPI_Type_commit(&(l->column_type));

    MPI_Type_vector(end[1] - begin[1], 1, l->nx, MPI_INT, &(l->column_type));
    MPI_Type_commit(&(l->column_type));

//      int size;
//    MPI_Aint ext;
//    MPI_Type_size(l->block_type, &size);
//    MPI_Type_extent(l->block_type, &ext);
//    printf("size = %d, ext = %d\n", size, ext);
}

void life_free(life_t *l) {
    free(l->u0);
    free(l->u1);
    l->nx = l->ny = 0;
    MPI_Type_free(&(l->block_type));
}

void life_save_vtk(const char *path, life_t *l) {
    FILE *f;
    int i1, i2, j;
    f = fopen(path, "w");
    assert(f);
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "Created by write_to_vtk2d\n");
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");
    fprintf(f, "DIMENSIONS %d %d 1\n", l->nx + 1, l->ny + 1);
    fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
    fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
    fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);

    fprintf(f, "SCALARS life int 1\n");
    fprintf(f, "LOOKUP_TABLE life_table\n");
    for (i2 = 0; i2 < l->ny; ++i2) {
        for (i1 = 0; i1 < l->nx; ++i1) {
            fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
        }
    }
    fclose(f);
}

void life_step(life_t *l) {
    int i, j;
    for (j = l->begin_k[1]; j < l->end_k[1]; ++j) {
        for (i = l->begin_k[0]; i < l->end_k[0]; ++i) {
            int n = 0;
            n += l->u0[ind(i + 1, j)];
            n += l->u0[ind(i + 1, j + 1)];
            n += l->u0[ind(i, j + 1)];
            n += l->u0[ind(i - 1, j)];
            n += l->u0[ind(i - 1, j - 1)];
            n += l->u0[ind(i, j - 1)];
            n += l->u0[ind(i - 1, j + 1)];
            n += l->u0[ind(i + 1, j - 1)];
            l->u1[ind(i, j)] = 0;
            if (n == 3 && l->u0[ind(i, j)] == 0) {
                l->u1[ind(i, j)] = 1;
            }
            if ((n == 3 || n == 2) && l->u0[ind(i, j)] == 1) {
                l->u1[ind(i, j)] = 1;
            }
            /* Проверка сбора данных. */
            // l->u1[ind(i, j)] = l->rank;
        }
    }
    int *tmp;
    tmp = l->u0;
    l->u0 = l->u1;
    l->u1 = tmp;
}

void life_decomposition(const int p, const int k, const int n, int *begin, int *end) {
    *begin = k * (n / p);
    *end = *begin + (n / p);
    if (k == p - 1) *end = n;
}

void life_collect(life_t *l) {
    /* Сбор по строкам. */
    if (l->coords[0] == l->dims[0] - 1) {
        int i;
        for (i = 0; i < l->dims[0] - 1; ++i) {
            int begin_i, end_i;
            life_decomposition(l->dims[0], i, l->nx, &begin_i, &end_i);
            MPI_Recv(l->u0 + ind(begin_i, l->begin_k[1]), 1, l->block_type, i, 0, l->dims_comm[1], MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(l->u0 + ind(l->begin_k[0], l->begin_k[1]), 1, l->block_type, l->dims[0] - 1, 0, l->dims_comm[1]);
    }
    /* Сбор по столбцам. */
    if (l->coords[0] == l->dims[0] - 1) {
        if (l->coords[1] == l->dims[1] - 1) {
            int i;
            for (i = 0; i < l->dims[1] - 1; ++i) {
                int begin_i, end_i;
                life_decomposition(l->dims[1], i, l->ny, &begin_i, &end_i);
                MPI_Recv(l->u0 + ind(0, begin_i), l->nx * (end_i - begin_i), MPI_INT, i, 0, l->dims_comm[0],
                         MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(l->u0 + ind(0, l->begin_k[1]), l->nx * (l->end_k[1] - l->begin_k[1]), MPI_INT, l->dims[1] - 1, 0,
                     l->dims_comm[0]);
        }
    }
}

void life_exchange(life_t *l) {
    if (l->size != 1) {
        int next[2] = {(l->dims[0] + l->coords[0] + 1) % l->dims[0], (l->dims[1] + l->coords[1] + 1) % l->dims[1]};
        int prev[2] = {(l->dims[0] + l->coords[0] - 1) % l->dims[0], (l->dims[1] + l->coords[1] - 1) % l->dims[1]};

        MPI_Send(l->u0 + ind(l->begin_k[0], l->begin_k[1]), l->end_k[0] - l->begin_k[0], MPI_INT, (l->dims[1] + l->coords[1] - 1) % l->dims[1], 0, l->dims_comm[0]);
    	MPI_Send(l->u0 + ind(l->begin_k[0], l->end_k[1] - 1), l->end_k[0] - l->begin_k[0], MPI_INT, (l->dims[1] + l->coords[1] + 1) % l->dims[1], 1, l->dims_comm[0]);
    	MPI_Recv(l->u0 + ind(l->begin_k[0], l->end_k[1]), l->end_k[0] - l->begin_k[0], MPI_INT, (l->dims[1] + l->coords[1] + 1) % l->dims[1], 0, l->dims_comm[0], MPI_STATUS_IGNORE);
    	MPI_Recv(l->u0 + ind(l->begin_k[0], l->begin_k[1] - 1), l->end_k[0] - l->begin_k[0], MPI_INT, (l->dims[1] + l->coords[1] - 1) % l->dims[1], 1, l->dims_comm[0], MPI_STATUS_IGNORE);
        
        MPI_Send(l->u0 + ind(l->begin_k[0], l->begin_k[1]), 1, l->column_type, (l->dims[0] + l->coords[0] - 1) % l->dims[0], 0, l->dims_comm[1]);
	    MPI_Send(l->u0 + ind(l->end_k[0] - 1, l->begin_k[1]), 1, l->column_type, (l->dims[0] + l->coords[0] + 1) % l->dims[0], 1, l->dims_comm[1]);
    	MPI_Recv(l->u0 + ind(l->end_k[0], l->begin_k[1]), 1, l->column_type, (l->dims[0] + l->coords[0] + 1) % l->dims[0], 0, l->dims_comm[1], MPI_STATUS_IGNORE);
    	MPI_Recv(l->u0 + ind(l->begin_k[0] - 1, l->begin_k[1]), 1, l->column_type, (l->dims[0] + l->coords[0] - 1) % l->dims[0], 1, l->dims_comm[1], MPI_STATUS_IGNORE);

        int left_down_rank, left_up_rank, right_down_rank, right_up_rank;
        MPI_Cart_rank(l->cart_comm, prev, &(left_down_rank));
        MPI_Cart_rank(l->cart_comm, next, &(right_up_rank));

        int left_up_coords[2] = {prev[0], next[1]};
        int right_down_coords[2] = {next[0], prev[1]};
        MPI_Cart_rank(l->cart_comm, left_up_coords, &(left_up_rank));
        MPI_Cart_rank(l->cart_comm, right_down_coords, &(right_down_rank));

        MPI_Send(l->u0 + ind(l->begin_k[0], l->begin_k[1]), 1, MPI_INT, left_down_rank, 0, l->cart_comm);
        MPI_Send(l->u0 + ind(l->end_k[0] - 1, l->end_k[1] - 1), 1, MPI_INT, right_up_rank, 1, l->cart_comm);
        MPI_Send(l->u0 + ind(l->begin_k[0], l->end_k[1] - 1), 1, MPI_INT, left_up_rank, 2, l->cart_comm);
        MPI_Send(l->u0 + ind(l->end_k[0] - 1, l->begin_k[1]), 1, MPI_INT, right_down_rank, 3, l->cart_comm);

        MPI_Recv(l->u0 + ind(l->begin_k[0] - 1, l->begin_k[1] - 1), 1, MPI_INT, left_down_rank, 1, l->cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(l->u0 + ind(l->end_k[0], l->end_k[1]), 1, MPI_INT, right_up_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(l->u0 + ind(l->begin_k[0] - 1, l->end_k[1]), 1, MPI_INT, left_up_rank, 3, l->cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(l->u0 + ind(l->end_k[0], l->begin_k[1] - 1), 1, MPI_INT, right_down_rank, 2, l->cart_comm, MPI_STATUS_IGNORE);
    }
}
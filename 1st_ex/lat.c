#include <cstdlib>
#include <string>
#include <math.h>
#include <mpi.h>
 
using namespace std; 
 
int main(int argc, char **argv) 
{ 
	MPI_Init(&argc, &argv); 
	int rank, m_size = (int) pow(10.0, (atoi(argv[1]) - 1) / 5.0); 
	if (atoi(argv[1]) == 37) { 
		m_size = 12000000; 
	} 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	char buf[m_size]; 

	if (rank == 0) { 
		string msg = string(m_size, 'a'); 
		strcpy(buf, msg.c_str()); 
		MPI_Send(buf, m_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD); 
	} else { 
		cout « m_size « ' '; 
		MPI_Recv(buf, m_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

	} 

	MPI_Finalize(); 
	return 0; 
}
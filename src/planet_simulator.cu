#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include "helpers.h"

const int X_MAX = 1000;
const int Y_MAX = 1000;
const int X_MIN = 0;
const int Y_MIN = 0;


int  n = 0;
int  t = 0;
double dt = 0;
std::string output_file_name;
bool use_cpu = false;
bool use_gpu = false;


// CPU side
//  array of x coordinates of the planets
double* x = nullptr;
//  array of y coordinates of the planets
double* y = nullptr;
//  array of x velocities of the planets
double* vx = nullptr;
//  array of y velocities of the planets
double* vy = nullptr;
// GPU side
//  array of x coordinates of the planets
double* d_x = nullptr;
//  array of y coordinates of the planets
double* d_y = nullptr;
//  array of x velocities of the planets
double* d_vx = nullptr;
//  array of y velocities of the planets
double* d_vy = nullptr;


int parse_arguments(int argc, char* argv[]) {
	// Arguments are:
	// 1. Number of planets
	// 2. Number of time steps
	// 3. Time step size
	// 4. Output file name
	// 5. Use CPU
	// 6. Use GPU
	// Check if the number of arguments is correct
	if (argc != 7) {
		std::cerr << "Usage: " << argv[0] << " <number of planets> <number of time steps> <time step size> <output file name> <use CPU> <use GPU>" << std::endl;
		return 1;
	}
	// Parse the arguments
	n = std::atoi(argv[1]);
	t = std::atoi(argv[2]);
	dt = std::atof(argv[3]);
	output_file_name = argv[4];
	use_cpu = std::atoi(argv[5]);
	use_gpu = std::atoi(argv[6]);

	// Check if the number of planets is correct
	if (n < 2) {
		std::cerr << "The number of planets must be at least 2" << std::endl;
		return 1;
	}
	// Check if the number of time steps is correct
	if (t < 1) {
		std::cerr << "The number of time steps must be at least 1" << std::endl;
		return 1;
	}
	// Check if the time step size is correct
	if (dt <= 0) {
		std::cerr << "The time step size must be positive" << std::endl;
		return 1;
	}

	return 0;
}

// thread function for updating the velocities of the planets
void  update_velocities_cpu(int index, double* x, double* y, double* vx, double* vy, double dt) {
	// Update the velocities of the planet
	for (int j = 0; j < n; j++) {
		if (index != j) {
			double dx = x[j] - x[index];
			double dy = y[j] - y[index];
			double d = std::sqrt(dx * dx + dy * dy);
			double f = 1.0 / (d * d);
			vx[index] += f * dx * dt;
			vy[index] += f * dy * dt;
		}
	}
}

// thread function for updating the positions of the planets
void update_positions_cpu(int index, double* x, double* y, double* vx, double* vy, double dt) {
	// Update the position of the planet
	x[index] += vx[index] * dt;
	y[index] += vy[index] * dt;
}

void simulate_cpu(int n, int t, double dt, const std::string& output_file_name) {
	// Open the output file
	std::ofstream output_file(output_file_name);
	if (!output_file.is_open()) {
		std::cerr << "Failed to open the output file" << std::endl;
		return;
	}
	// Initialize the planets
	double* x = new double[n];
	double* y = new double[n];
	double* vx = new double[n];
	double* vy = new double[n];
	for (int i = 0; i < n; i++) {
		x[i] = 1.0 * i;
		y[i] = 0.0;
		vx[i] = 0.0;
		vy[i] = 0.0;
	}
	// Simulate the planets
	std::thread* threads = new std::thread[n];
	for (int i = 0; i < t; i++) {
		// Output the positions of the planets
		for (int j = 0; j < n; j++) {
			output_file << x[j] << " " << y[j] << " ";
		}
		output_file << std::endl;
		// update the velocities of the planets using threads
		// create one thread per planet
		for (int j = 0; j < n; j++) {
			threads[j] = std::thread(update_velocities_cpu, j, x, y, vx, vy, dt);
		}
		// wait for all threads to finish
		for (int j = 0; j < n; j++) {
			threads[j].join();
		}
		// update the positions of the planets using threads
		// create one thread per planet
		for (int j = 0; j < n; j++) {
			threads[j] = std::thread(update_positions_cpu, j, x, y, vx, vy, dt);
		}
		// wait for all threads to finish
		for (int j = 0; j < n; j++) {
			threads[j].join();
		}
	}
	// deallocate the threads
	delete[] threads;
	// Close the output file
	output_file.close();
	// Deallocate the arrays
	delete[] x;
	delete[] y;
	delete[] vx;
	delete[] vy;
}

void init_arrays_cpu() {
	debug_print("CPU: init arrays");
	// Allocate the arrays
	x = new double[n];
	y = new double[n];
	vx = new double[n];
	vy = new double[n];
	// Initialize the planets
	for (int i = 0; i < n; i++) {
		x[i] = 1.0 * i;
		y[i] = 0.0;
		vx[i] = 0.0;
		vy[i] = 0.0;
	}
	debug_print("CPU: arrays initialized");
}

void free_arrays_cpu() {
	// Deallocate the arrays
	delete[] x;
	delete[] y;
	delete[] vx;
	delete[] vy;
}

void init_arrays_gpu() {
	debug_print("GPU: init arrays");
	// Allocate the arrays
	cudaMalloc(&d_x, n * sizeof(double));
	cudaMalloc(&d_y, n * sizeof(double));
	cudaMalloc(&d_vx, n * sizeof(double));
	cudaMalloc(&d_vy, n * sizeof(double));
	debug_print("GPU: arrays initialized");
}

void copy_arrays_to_gpu() {
	// Copy the arrays to the GPU
	cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx, vx, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, vy, n * sizeof(double), cudaMemcpyHostToDevice);
}

void copy_arrays_from_gpu() {
	// Copy the arrays from the GPU
	cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(vx, d_vx, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(vy, d_vy, n * sizeof(double), cudaMemcpyDeviceToHost);
}

void free_arrays_gpu() {
	// Deallocate the arrays
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_vx);
	cudaFree(d_vy);
}

__global__ void update_velocities(int n, double* d_x, double* d_y, double* d_vx, double* d_vy, double dt) {
	// Get the index of the planet
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// Check if the index is valid
	if (i < n) {
		// Update the velocities of the planet
		for (int j = 0; j < n; j++) {
			if (i != j) {
				double dx = d_x[j] - d_x[i];
				double dy = d_y[j] - d_y[i];
				double d = sqrt(dx * dx + dy * dy);
				double f = 1.0 / (d * d);
				d_vx[i] += f * dx * dt;
				d_vy[i] += f * dy * dt;
			}
		}
	}
}

__global__ void update_positions(int n, double* d_x, double* d_y, double* d_vx, double* d_vy, double dt) {
	// Get the index of the planet
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// Check if the index is valid
	if (i < n) {
		// Update the position of the planet
		d_x[i] += d_vx[i] * dt;
		d_y[i] += d_vy[i] * dt;
	}
}

void simulate_gpu(int n, int t, double dt, const std::string& output_file_name) {
	// init arrays
	init_arrays_gpu();
	init_arrays_cpu();
	// Copy the arrays to the GPU
	copy_arrays_to_gpu();

	// Open the output file
	debug_print("GPU: open output file");
	std::ofstream output_file(output_file_name);
	if (!output_file.is_open()) {
		std::cerr << "Failed to open the output file" << std::endl;
		return;
	}
	debug_print("GPU: output file opened");
	// Simulate the planets
	for (int i = 0; i < t; i++) {
		// Copy the arrays from the GPU
		copy_arrays_from_gpu();
		// Output the positions of the planets
		for (int j = 0; j < n; j++) {
			output_file << x[j] << " " << y[j] << " ";
		}
		output_file << std::endl;
		// Update the velocities of the planets
		update_velocities<<<(n + 255) / 256, 256>>>(n, d_x, d_y, d_vx, d_vy, dt);
		//cudaDeviceSynchronize();
		// Update the positions of the planets
		update_positions<<<(n + 255) / 256, 256>>>(n, d_x, d_y, d_vx, d_vy, dt);
		//cudaDeviceSynchronize();
	}

}


int main(int argc, char* argv[]) {
	// Parse the arguments
	if (parse_arguments(argc, argv) != 0) {
		return 1;
	}
	// Simulate the planets with CPU
	// Add to output file name the cpu prefix
	time_t start, end;
	if (use_cpu) {
		std::cout << "CPU" << std::endl;
		output_file_name = "cpu_" + output_file_name;
		// Compute time for CPU in nanoseconds
		time(&start);
		simulate_cpu(n, t, dt, output_file_name);
		time(&end);
		std::cout << "CPU time: " << difftime(end, start) << " seconds" << std::endl;
	}
	if (use_gpu) {
		// Simulate the planets with GPU
		// Add to output file name the gpu prefix
		std::cout << "GPU" << std::endl;
		output_file_name = "gpu_" + output_file_name;
		// Compute time for GPU
		time(&start);
		simulate_gpu(n, t, dt, output_file_name);
		time(&end);
		std::cout << "GPU time: " << difftime(end, start) << " seconds" << std::endl;
	}


	return 0;
}

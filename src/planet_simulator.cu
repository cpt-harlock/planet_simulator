#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <pthread.h>
#include <SFML/Graphics.hpp>
#include <signal.h>
#include "helpers.h"

const int X_MAX = 1920;
const int Y_MAX = 1080;
const int X_MIN = 0;
const int Y_MIN = 0;

// Upper and lower bounds for the size of the planets
// Use for max radius Jupiter
const double MAX_RADIUS = 69911.0e3;
// Use for min radius Mercury
const double MIN_RADIUS = 2439.7e3;

const int MIN_SIZE = 1;
const int MAX_SIZE = 20;

const double G = 6.67430e-11;

// Use same average density for all planets as Earth
const double DENSITY = 5514.0;


int  n = 0;
int  t = 0;
double dt = 0;
std::string output_file_name;
bool use_cpu = false;
bool use_gpu = false;
bool infinite_loop = false;

// Thread barrier
// TODO: later for threads that are not killed and respawned between time steps
pthread_barrier_t* barrier;

// FPS counter
int fps = 0;

// Iterations per second counter
int ips = 0;

// Elasped iterations
int iterations = 0;

// Drawing counter
int draw = 0;


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


struct color {
	int r;
	int g;
	int b;
} ;

struct color* colors = nullptr;

// arrays for planet sizes
int* sizes = nullptr;
int* d_sizes = nullptr;

void init_planet_positions() {
	// Initialize the positions of the planets
	for (int i = 0; i < n; i++) {
		x[i] = X_MIN + (X_MAX - X_MIN) * (rand() / (RAND_MAX + 1.0));
		y[i] = Y_MIN + (Y_MAX - Y_MIN) * (rand() / (RAND_MAX + 1.0));
	}
}

void init_planet_velocities() {
	// Initialize the velocities of the planets
	for (int i = 0; i < n; i++) {
		vx[i] = 0;
		vy[i] = 0;
	}
}

// Thread function that compute and update FPS
void compute_ips() {
	while (true) {
		// Store the number of iterations
		int current_iterations = iterations;
		// Sleep for one second
		std::this_thread::sleep_for(std::chrono::seconds(1));
		// Compute the IPS
		ips = iterations - current_iterations;
	}
}

// Thread function that compute and update FPS
void compute_fps() {
	while (true) {
		// Store the number of draws
		int current_draw = draw;
		// Sleep for one second
		std::this_thread::sleep_for(std::chrono::seconds(1));
		// Compute the FPS
		fps = draw - current_draw;
	}
}

void init_barriers() {
	// Initialize the barrier
	barrier = new pthread_barrier_t;
	pthread_barrier_init(barrier, nullptr, n);
}

void init_colors() {
	// Initialize the colors of the planets
	colors = new struct color[n];
	for (int i = 0; i < n; i++) {
		colors[i].r = rand() % 256;
		colors[i].g = rand() % 256;
		colors[i].b = rand() % 256;
	}
}

void init_sizes() {
	// Initialize the sizes of the planets randomly between MIN_SIZE and MAX_SIZE
	sizes = new int[n];
	for (int i = 0; i < n; i++) {
		sizes[i] = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * (rand() / (RAND_MAX + 1.0));
	}
	// Allocate the sizes on the GPU
	cudaMalloc(&d_sizes, n * sizeof(int));
	cudaMemcpy(d_sizes, sizes, n * sizeof(int), cudaMemcpyHostToDevice);
}


int parse_arguments(int argc, char* argv[]) {
	// Arguments are:
	// 1. Number of planets
	// 2. Number of time steps
	// 3. Time step size
	// 4. Output file name
	// 5. Use CPU
	// 6. Use GPU
	// 7. Infinite loop

	// Check if the number of arguments is correct
	if (argc != 8) {
		std::cerr << "Usage: " << argv[0] << " <number of planets> <number of time steps> <time step size> <output file name> <use CPU> <use GPU> <infinite loop>" << std::endl;
		return 1;
	}
	// Parse the arguments
	n = std::atoi(argv[1]);
	t = std::atoi(argv[2]);
	dt = std::atof(argv[3]);
	output_file_name = argv[4];
	use_cpu = std::atoi(argv[5]);
	use_gpu = std::atoi(argv[6]);
	infinite_loop = std::atoi(argv[7]);

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
			double m1 = DENSITY * 4.0 / 3.0 * M_PI * pow(sizes[index], 3);
			double m2 = DENSITY * 4.0 / 3.0 * M_PI * pow(sizes[j], 3);
			double f = G * m1 * m2 / (d * d * d);
			double a = f / m1;
			vx[index] += a * dx * dt;
			vy[index] += a * dy * dt;
		}
	}
}


// thread function for updating the positions of the planets
void update_positions_cpu(int index, double* x, double* y, double* vx, double* vy, double dt) {
	// Update the position of the planet
	x[index] += vx[index] * dt;
	y[index] += vy[index] * dt;
	// print  the position of the planet
	//std::cout << "Planet " << index << " x: " << x[index] << " y: " << y[index] << std::endl;
	// Check if the planet is out of bounds
	if (x[index] < X_MIN || x[index] > X_MAX || y[index] < Y_MIN || y[index] > Y_MAX) {
		std::cerr << "Planet " << index << " is out of bounds" << std::endl;
		// Make  the planet reappear on the other side of the screen
		x[index] = X_MIN + (X_MAX - X_MIN) * (rand() / (RAND_MAX + 1.0));
		y[index] = Y_MIN + (Y_MAX - Y_MIN) * (rand() / (RAND_MAX + 1.0));
	}

}

void simulate_cpu(int n, int t, double dt, const std::string& output_file_name) {
	// Reset iterations
	iterations = 0;
	// Reset planet positions
	init_planet_positions();
	// Reset planet velocities
	init_planet_velocities();
	// Open the output file
	std::ofstream output_file(output_file_name);
	if (!output_file.is_open()) {
		std::cerr << "Failed to open the output file" << std::endl;
		return;
	}
	// Simulate the planets
	std::thread* threads = new std::thread[n];
	int i = 0;
	for (i = 0; ; i++) {
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
		// increment the iterations
		iterations++;
		if (!infinite_loop && i >= t - 1) {
			break;
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

__global__ void update_velocities(int n, double* d_x, double* d_y, double* d_vx, double* d_vy, double dt, int* d_sizes) {
	// Get the index of the planet
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// Check if the index is valid
	if (i < n) {
		// Update the velocities of the planet
		for (int j = 0; j < n; j++) {
			if (i != j) {
				// Compute the distance between the planets
				double dx = d_x[j] - d_x[i];
				double dy = d_y[j] - d_y[i];
				double d = sqrt(dx * dx + dy * dy);
				// Compute planet masses
				double m1 = DENSITY * 4.0 / 3.0 * M_PI * pow(d_sizes[i], 3);
				double m2 = DENSITY * 4.0 / 3.0 * M_PI * pow(d_sizes[j], 3);
				// Compute the force between the planets using Newton's law of universal gravitation, F = G * m1 * m2 / d^2
				double f = G * m1 * m2 / (d * d * d);
				double a = f / m1;
				// Update the velocity of the planet
				d_vx[i] += a * dx * dt;
				d_vy[i] += a * dy * dt;

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
	// Check if the planet is out of bounds
	if (d_x[i] < X_MIN || d_x[i] > X_MAX || d_y[i] < Y_MIN || d_y[i] > Y_MAX) {
		// Make the planet reappear on the other side of the screen
		if (d_x[i] < X_MIN) {
			d_x[i] = X_MAX;
		}
		else if (d_x[i] > X_MAX) {
			d_x[i] = X_MIN;
		}
		if (d_y[i] < Y_MIN) {
			d_y[i] = Y_MAX;
		}
		else if (d_y[i] > Y_MAX) {
			d_y[i] = Y_MIN;
		}
	}
}

void simulate_gpu(int n, int t, double dt, const std::string& output_file_name) {
	// Reset iterations
	iterations = 0;
	// Reset planet positions
	init_planet_positions();
	// Reset planet velocities
	init_planet_velocities();

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
	int i = 0;
	for (i = 0; ; i++) {
		// Copy the arrays from the GPU
		copy_arrays_from_gpu();
		// Output the positions of the planets
		for (int j = 0; j < n; j++) {
			output_file << x[j] << " " << y[j] << " ";
		}
		output_file << std::endl;
		// Update the velocities of the planets
		update_velocities<<<(n + 255) / 256, 256>>>(n, d_x, d_y, d_vx, d_vy, dt, d_sizes);
		//cudaDeviceSynchronize();
		// Update the positions of the planets
		update_positions<<<(n + 255) / 256, 256>>>(n, d_x, d_y, d_vx, d_vy, dt);
		//cudaDeviceSynchronize();
		// Increment the iterations
		iterations++;
		if (!infinite_loop && i >= t - 1) {
			break;
		}
	}

}

void simulation() {
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
}


void draw_planets(sf::RenderWindow& window) {
	// Draw the planets with  random colors
	for (int i = 0; i < n; i++) {
		// Map planet size to radius using linear interpolation
		// 1. Find the percentage of the size of the planet between MIN_SIZE and MAX_SIZE
		double percentage = (sizes[i] - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS);
		// 2. Find the radius of the planet between MIN_RADIUS and MAX_RADIUS
		double radius = MIN_SIZE + percentage * (MAX_SIZE - MIN_SIZE);
		// 3. Draw the planet
		sf::CircleShape planet(radius);
		planet.setFillColor(sf::Color(colors[i].r, colors[i].g, colors[i].b));
		planet.setPosition(x[i], y[i]);
		window.draw(planet);
	}
}

void init_memory() {
	init_arrays_cpu();
	init_arrays_gpu();
}

void init_planets() {
	init_planet_positions();
	init_planet_velocities();
	init_colors();
	init_sizes();
}

void init_rand_seed() {
	// Initialize the random seed
	srand(time(nullptr));
}

int main(int argc, char* argv[]) {
	// Parse the arguments
	if (parse_arguments(argc, argv) != 0) {
		return 1;
	}

	// Initialize the random seed
	init_rand_seed();

	// Create the window the same size as the screen
	sf::RenderWindow window(sf::VideoMode(X_MAX, Y_MAX), "Planet Simulator");


	// Load the font
	sf::Font font;
	if (!font.loadFromFile("/usr/share/fonts/truetype/msttcorefonts/arial.ttf")) {
		std::cerr << "Failed to load the font" << std::endl;
		return 1;
	}
	// Create the text
	sf::Text text;
	sf::Text text2;
	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::White);
	text.setPosition(10, 10);
	text2.setFont(font);
	text2.setCharacterSize(24);
	text2.setFillColor(sf::Color::White);
	text2.setPosition(10, 100);


	// Initialize memory
	init_memory();

	// Initialize the planets
	init_planets();

	// Start a thread for the simulation
	std::thread simulation_thread(simulation);

	// Start a thread for computing the FPS
	std::thread ips_thread(compute_ips);

	// Start a thread for computing the IPS
	std::thread fps_thread(compute_fps);

	// Register signal handler for SIGINT (Ctrl+C)
	signal(SIGINT, [](int signum) {
		// Exit
		exit(0);
	});

	// Start the game loop
	while (window.isOpen()) {
		// Process events
		sf::Event event;
		while (window.pollEvent(event))
		{
			// Close window: exit
			if (event.type == sf::Event::Closed)
				window.close();
		}

		// Clear screen
		window.clear();

		// Draw the string
		window.draw(text);

		// Draw the planets
		draw_planets(window);


		// Update the window
		window.display();

		// Increment the draw counter
		draw++;

		// Write FPS and IPS to the text
		text.setString("FPS: " + std::to_string(fps) + " IPS: " + std::to_string(ips));
		window.draw(text);

	}

	return 0;
}

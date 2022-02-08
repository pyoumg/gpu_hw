#pragma warning(disable: 4996)
//
//  OpenGL_WE.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright ©2021 Sogang University. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>

#include "Util/my_OpenCL_util_2_2.h"
#include "Context_WE.h"
#include "Config_WE.h"

#define BUFFER_OFFSET(offset) ((GLvoid *) (offset))
#define INDEX_VERTEX_POSITION 0

#include "Shaders/LoadShaders.h"
GLuint h_ShaderProgram; // handle to shader program
GLint loc_ModelViewProjectionMatrix, loc_primitive_color; // indices of uniform variables

void prepare_shader_program(void) {
	ShaderInfo shader_info[3] = {
		{ GL_VERTEX_SHADER, "./Source/Shaders/simple.vert" },
		{ GL_FRAGMENT_SHADER, "./Source/Shaders/simple.frag" },
		{ GL_NONE, NULL }
	};

	h_ShaderProgram = LoadShaders(shader_info);
	glUseProgram(h_ShaderProgram);
	loc_ModelViewProjectionMatrix = glGetUniformLocation(h_ShaderProgram, "u_ModelViewProjectionMatrix");
	loc_primitive_color = glGetUniformLocation(h_ShaderProgram, "u_primitive_color");
	glUniform3f(loc_primitive_color, 255/255.0f, 87/255.0f, 51/255.0f);
	glUseProgram(0);
}

void initialize_grid_data(CONTEXT_WE* cont) {
	WAVE_EQUATION* we = &cont->wave_equation;

	float* gr_0 = we->grid_data[0];
	float* gr_1 = we->grid_data[1];
	float* gr_2 = we->grid_data[2];
	float* gr_tmp = we->grid_data_tmp;

	int n_node_1d = we->n_nodes_1d;
	float sigma_squared = we->sigma * we->sigma;
	float h = we->side_length / we->n_nodes_1d;

	int index = 0;
	int x_mid = we->x_mid, y_mid = we->y_mid;
	for (int j = 0; j < n_node_1d; j++) {
		for (int i = 0; i < n_node_1d; i++) {
			gr_0[4 * index + 0] = i * h; // x
			gr_0[4 * index + 1] = we->amplitude * (pow(M_E, (-1) * (h*h*((i - x_mid) * (i - x_mid) + (j - y_mid) * (j - y_mid)))
				/ (2 * sigma_squared))) / (2 * M_PI * sigma_squared) 
				* we->init_speed * we->time_interval;  // y
			gr_0[4 * index + 2] = j * h; // z
			gr_0[4 * index + 3] = 1.0f;  // w

			gr_1[4 * index + 0] = i * h; // x
			gr_1[4 * index + 1] = 0.0f;  // y
			gr_1[4 * index + 2] = j * h; // z
			gr_1[4 * index + 3] = 1.0f;  // w

			gr_2[4 * index + 0] = i * h; // x
			gr_2[4 * index + 1] = 0.0f;  // y
			gr_2[4 * index + 2] = j * h; // z
			gr_2[4 * index + 3] = 1.0f;  // w

			gr_tmp[4 * index + 0] = i * h; // x
			gr_tmp[4 * index + 1] = 0.0f;  // y
			gr_tmp[4 * index + 2] = j * h; // z
			gr_tmp[4 * index + 3] = 1.0f;  // w

			index++;
		}
	}
}

void prepare_grids(CONTEXT_WE* cont) {
	WAVE_EQUATION* we = &cont->wave_equation;
	OGL_STUFFS* ogl = &cont->ogl_stuffs;

	int n_nodes_1d = we->n_nodes_1d;

	for (int i = 0; i < 3; i++) {
		we->grid_data[i] = (float*)malloc(4 * sizeof(float) * n_nodes_1d * n_nodes_1d);
	}
	if (!we->grid_data[0] || !we->grid_data[1] || !we->grid_data[2]) {
		fprintf(stderr, "*** Error: cannot allocate memory for grid_data[]...\n");
		exit(-1);
	}
	we->grid_data_tmp = (float*)malloc(4 * sizeof(float) * n_nodes_1d * n_nodes_1d);
	if (!we->grid_data_tmp) {
		fprintf(stderr, "*** Error: cannot allocate memory for grid_data_tmp...\n");
		exit(-1);
	}
	initialize_grid_data(cont);

	glGenBuffers(3, ogl->grid_data_OGL_BO);
	for (int i = 0; i < 3; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, ogl->grid_data_OGL_BO[i]);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * n_nodes_1d * n_nodes_1d, we->grid_data[i], 
			GL_DYNAMIC_DRAW);
	}

	GLuint* grid_index = ogl->grid_indices = (GLuint*)malloc(sizeof(GLuint) * (n_nodes_1d - 1) * (2 * n_nodes_1d + 1));
	if (!grid_index) {
		fprintf(stderr, "*** Error: cannot allocate memory for grid indices...\n");
		exit(-1);
	}
	ogl->n_grid_indices = (n_nodes_1d - 1) * (2 * n_nodes_1d + 1);
	int grid_index_id = 0;

	for (int i = 0; i < n_nodes_1d - 1; i++) {
		for (int j = 0; j < n_nodes_1d; j++) {
			grid_index[grid_index_id++] = i * n_nodes_1d + j;
			grid_index[grid_index_id++] = (i + 1) * n_nodes_1d + j;
		}
		grid_index[grid_index_id++] = 65536;
	}

	if (grid_index_id != ogl->n_grid_indices) {
		fprintf(stderr, "^^^ Error: something wrong happened during the index array generation...\n");
		exit(-1);
	}

	//element indices
	glGenBuffers(1, &ogl->grid_index_OGL_BO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl->grid_index_OGL_BO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * ogl->n_grid_indices,
		ogl->grid_indices, GL_DYNAMIC_COPY);

	// Initialize vertex array object.
	glGenVertexArrays(3, ogl->VAO);
 	for (int i = 0; i < 3; i++) {
		glBindVertexArray(ogl->VAO[i]);

		glBindBuffer(GL_ARRAY_BUFFER, ogl->grid_data_OGL_BO[i]);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
		glEnableVertexAttribArray(INDEX_VERTEX_POSITION);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl->grid_index_OGL_BO);
 	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// Begin of Callback function definitions
#define TO_RADIAN 0.01745329252f  
#define TO_DEGREE 57.295779513f

// include glm/*.hpp only if necessary
// #include <glm/glm.hpp> 
#include <glm/gtc/matrix_transform.hpp> //translate, rotate, scale, lookAt, perspective, etc.
glm::mat4 ViewProjectionMatrix, ViewMatrix, ProjectionMatrix;
glm::mat4 ModelViewProjectionMatrix; // This one is sent to vertex shader when ready.

/*  Callback functions */
static CONTEXT_WE* context_OGL; // for callback functions only

static float fovys[5] = { 25.0f, 20.0f, 15.0f, 10.0f, 5.0f };
static int fovys_index = 0;
static int flag_simulation = 0;

#if defined USE_GPU_EXPLICIT
const char simulation_method_name[32] = "GPU_EXPLICIT";
#elif defined USE_GPU_IMPLICIT_JACOBI
const char simulation_method_name[32] = "GPU_IMPLICIT_JACOBI";
#elif defined USE_CPU_IMPLICIT_JACOBI
const char simulation_method_name[32] = "CPU_IMPLICIT_JACOBI";
#elif defined USE_CPU_IMPLICIT_GAUSS_SEIDEL
const char simulation_method_name[32] = "CPU_IMPLICIT_GAUSS_SEIDEL";
#endif

void display(void) {
	static int simulation_steps = 0;
	static float total_time = 0.0f;
	float time_per_frame, fps;
	char buf[256];

	if (flag_simulation) {
		if (simulation_steps < context_OGL->wave_equation.simulation_steps) {
			__int64 _start, _freq, _end;
			float _time;
			CHECK_TIME_START(_start, _freq);

#if defined USE_GPU_EXPLICIT
			run_CL_KERNEL_WE_EXPLICIT(context_OGL); 
#elif defined USE_GPU_IMPLICIT_JACOBI
			run_CL_KERNEL_WE_IMPLICIT(context_OGL);  
#elif defined USE_CPU_IMPLICIT_JACOBI
			run_CPU_Jacobi_WE(context_OGL);
#elif defined USE_CPU_IMPLICIT_GAUSS_SEIDEL
			run_CPU_GaussSeidel_WE(context_OGL);
#endif
			CHECK_TIME_END(_start, _end, _freq, _time);
			total_time += _time; //printf("total_time = %f\n", total_time);
			simulation_steps++;

			if (simulation_steps % 100 == 0) {
				time_per_frame = total_time / simulation_steps;
				fps = 1000.0f / time_per_frame;
				fprintf(stdout, "\n^^^ The average simulation time per frame after %d steps is %.3f ms(%.2f fps).\n",
					simulation_steps, time_per_frame, fps);
				sprintf(buf, "hw4_20191571: %.2f fps on average after %d steps by %s\n",
					fps, simulation_steps, simulation_method_name);
				glutSetWindowTitle(buf);
			}
		}
		else {
			time_per_frame = total_time / simulation_steps;
			fps = 1000.0f / time_per_frame;
			fprintf(stdout, "\n^^^ The average simulation time per frame after %d steps is %.3f ms(%.2f fps).\n",
				simulation_steps, time_per_frame, fps);
			sprintf(buf, "hw4_20191571: %.2f fps on average after %d steps\n",
				fps, simulation_steps);
			glutSetWindowTitle(buf);

			flag_simulation = 0;
			glUseProgram(h_ShaderProgram);
			glUniform3f(loc_primitive_color, 229/255.0f, 152/255.0f, 102/255.0f);
			glUseProgram(0);
		}
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(h_ShaderProgram);
	ModelViewProjectionMatrix = ViewProjectionMatrix;
	glUniformMatrix4fv(loc_ModelViewProjectionMatrix, 1, GL_FALSE, &ModelViewProjectionMatrix[0][0]);

    glBindVertexArray(context_OGL->ogl_stuffs.VAO[context_OGL->wave_equation.next_buffer]);
#ifdef  USE_CPU_IMPLICIT_JACOBI 
	glBindBuffer(GL_ARRAY_BUFFER, 
		context_OGL->ogl_stuffs.grid_data_OGL_BO[context_OGL->wave_equation.next_buffer]);
	int n_nodes_1d = context_OGL->wave_equation.n_nodes_1d;
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * n_nodes_1d * n_nodes_1d,
		context_OGL->wave_equation.grid_data[context_OGL->wave_equation.next_buffer], GL_DYNAMIC_DRAW);
#endif

#ifdef  USE_CPU_IMPLICIT_GAUSS_SEIDEL 
	glBindBuffer(GL_ARRAY_BUFFER,
		context_OGL->ogl_stuffs.grid_data_OGL_BO[context_OGL->wave_equation.next_buffer]);
	int n_nodes_1d = context_OGL->wave_equation.n_nodes_1d;
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * n_nodes_1d * n_nodes_1d,
		context_OGL->wave_equation.grid_data[context_OGL->wave_equation.next_buffer], GL_DYNAMIC_DRAW);
#endif

	glEnable(GL_PRIMITIVE_RESTART);
	glDrawElements(GL_TRIANGLE_STRIP, context_OGL->ogl_stuffs.n_grid_indices, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
	glDisable(GL_PRIMITIVE_RESTART);
 	glBindVertexArray(0);

	glutSwapBuffers();
}

void timer_scene(int timestamp_scene) {
	glutPostRedisplay();
 	if (flag_simulation)
		glutTimerFunc(context_OGL->wave_equation.display_interval, timer_scene, 0);
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 27: // ESC key
		glutLeaveMainLoop(); // Incur destuction callback for cleanups.
		break;
	case 's':
		flag_simulation = 1 - flag_simulation;
		if (flag_simulation) {
			fprintf(stdout, "$$$ Simulation started!\n");
			glutTimerFunc(context_OGL->wave_equation.display_interval, timer_scene, 0);
		}
		else {
			fprintf(stdout, "$$$ Simulation stopped!\n");
		}
		break;
	case 'z':
		float aspect_ratio = (float)context_OGL->window_width / context_OGL->window_height;
		fovys_index = (fovys_index + 1) % 5;
		ProjectionMatrix = glm::perspective(fovys[fovys_index] * TO_RADIAN, aspect_ratio, 1.0f, 1000.0f);

		ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;
		glutPostRedisplay();
	}
}

void reshape(int width, int height) {
	context_OGL->window_width = width; context_OGL->window_height = height;
	float aspect_ratio = (float)width / height;

	glViewport(0, 0, width, height);
	ProjectionMatrix = glm::perspective(fovys[fovys_index] * TO_RADIAN, aspect_ratio, 1.0f, 1000.0f);

	ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;
	glutPostRedisplay();
}

void clean_up_OpenCL(CONTEXT_WE* cont);
void clean_up_OpenGL(CONTEXT_WE* cont) {
	OGL_STUFFS* ogl_stuffs = &context_OGL->ogl_stuffs;
	if (!ogl_stuffs->grid_indices)
		free(ogl_stuffs->grid_indices);

	glDeleteBuffers(3, ogl_stuffs->grid_data_OGL_BO);
	glDeleteBuffers(1, &ogl_stuffs->grid_index_OGL_BO);
	glDeleteVertexArrays(3, ogl_stuffs->VAO);

	WAVE_EQUATION* we = &context_OGL->wave_equation;
	for (int i = 0; i < 3; i++) {
		if (!we->grid_data[i])
			free(we->grid_data[i]);
	}
	if (!we->grid_data_tmp)
		free(we->grid_data_tmp);
}

void cleanup(void) {
	// OpenCL context must be released before OpenGL context.
#if (defined USE_GPU_EXPLICIT) || (defined USE_GPU_IMPLICIT_JACOBI)
	clean_up_OpenCL(context_OGL);
#endif
	clean_up_OpenGL(context_OGL);
}

void register_callbacks(void) {
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutCloseFunc(cleanup);
}

void initialize_OpenGL_render(CONTEXT_WE* cont) {
	glClearColor(130/255.0f, 224/255.0f, 170/255.0f, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//glEnable(GL_CULL_FACE);

	glPrimitiveRestartIndex(65536);
	glLineWidth(1.0f);

	ViewMatrix = glm::lookAt(glm::vec3(200.0f, 90.0f, 200.0f),
		glm::vec3(cont->wave_equation.side_length / 2, 10.0f, cont->wave_equation.side_length / 2),
		glm::vec3(0.0f, 1.0f, 0.0f));

	fovys_index = 0;
	flag_simulation = 0;
}

void initialize_glew(void) {
	GLenum error;

	glewExperimental = GL_TRUE;
	error = glewInit();
	if (error != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(error));
		exit(-1);
	}
	fprintf(stdout, "*********************************************************\n");
	fprintf(stdout, " - GLEW version supported: %s\n", glewGetString(GLEW_VERSION));
	fprintf(stdout, " - OpenGL renderer: %s\n", glGetString(GL_RENDERER));
	fprintf(stdout, " - OpenGL version supported: %s\n", glGetString(GL_VERSION));
	fprintf(stdout, "*********************************************************\n\n");

}

void initialize_OpenGL(CONTEXT_WE* cont) {
	context_OGL = cont;

	initialize_glew();
	register_callbacks();
	prepare_grids(cont);
	prepare_shader_program();
	initialize_OpenGL_render(cont);
}
#include "data_types.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>

// Bad practice but this code is for learning about nns.
using namespace std;

class Network
{
public:

	// Each entry in layout is the number of neurons for that layer.
	// The first entry in layout is the input layer.
	vector<unsigned> layout;
	unsigned layers;

	// Layer->Neurons->Weights->value.
	vector<vector<vector<float>>> weights;

	// Layer->Neurons->value.
	vector<vector<float>> biases;

	// The step size when computing derivatives for gradient descent.
	float step = 0.0075;

	// Distance moved in gradient vector direction when learning
	float learning_rate;

	// Data used by the network when training.
	Training_Data td;

	Network(vector<unsigned> net_layout, float learning_rt)
	{
		assert(net_layout.size() > 1);
		layout = net_layout;
		learning_rate = learning_rt;
		layers = net_layout.size() - 1;

		// Set inital weights and biases using a normal distribution
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		default_random_engine generator(seed);
		normal_distribution<float> norm_dist(0.0, 1.0);

		// Start at layer 1 because layer 0 is the input and has no weights.
		for(int i = 1; i < layout.size(); i++)
		{
			vector<vector<float>> new_layer;
			vector<float> new_biases;
			
			// For each neuron, populate its weights and bias.
			for(int j = 0; j < layout[i]; j++)
			{
				vector<float> new_weights;
				new_biases.push_back(norm_dist(generator));

				// Neurons in the ith layer have as many weights as neurons in the i-1 layer.
				for(int k = 0; k < layout[i-1]; k++)
					new_weights.push_back(norm_dist(generator));

				new_layer.push_back(new_weights);
			}

			biases.push_back(new_biases);
			weights.push_back(new_layer);
		}
	}

	// Computes the output of the net for the given input recursively.
	// TODO try to pass inputs by reference to save cpu time.
	vector<float> output(vector<float> inputs, unsigned layer)
	{
		// Recursive call until the first layer is reached
		if(layer > 0)
			inputs = output(inputs, layer-1);
		
		vector<float> outputs;

		// Recall weights is structured: Layer->Neurons->Weights->value
		auto b_it = biases[layer].begin();
		for(const auto &wts : weights[layer])
		{
			outputs.push_back(sigmoid_output(inputs, wts, *b_it));
			b_it++;
		}

		return outputs;
	}

	// Returns the output of a sigmoid neuron with the passed params.
	float sigmoid_output(
		const vector<float> &inputs,
		const vector<float> &wts,
		const float b) 
	{
		return 1 / (1 + exp(-dot_vec(inputs, wts) - b));
	}

	// Mean Squared Error of outputs to expected outputs.
	float cost()
	{	
		vector<float> nn_outs;
		float err;
		float sum_of_squared_err;

		for(int i = 0; i < td.inputs.size(); i++)
		{
			nn_outs = output(td.inputs[i], layers-1);

			for(int j = 0; j < nn_outs.size(); j++)
			{
				err = td.outputs[i][j] - nn_outs[j];
				sum_of_squared_err += err*err;
			}
		}

		return sum_of_squared_err / (2*td.inputs.size());
	}

	// Returns gradient vector of the cost function wrt weights
	vector<float> unit_grad_cost_w()
	{
		float original_cost = cost();
		float original_weight;
		vector<float> grad;

		// Take the partial derivative wrt every weight in the net
		// The vector of these partials is the gradient vector.

		for(auto &layer : weights)
		{
			for(auto &neurons : layer)
			{
				for(auto &weight : neurons)
				{
					original_weight = weight;
					weight += step;
					grad.push_back((cost()-original_cost)/step);
					weight = original_weight;
				}
			}
		}

		make_unit_vector(grad);
		return grad;
	}

	// Returns gradient vector of the cost function wrt biases
	vector<float> unit_grad_cost_b()
	{
		float original_cost = cost();
		float original_bias;
		vector<float> grad;

		// Take the partial derivative wrt every bias in the net
		// The vector of these partials is the gradient vector.
		for(auto &layer : biases)
		{
			for(auto &bias : layer)
			{
				// Approx partial derivative
				original_bias = bias;
				bias += step;
				grad.push_back((cost()-original_cost)/step);
				bias = original_bias;
			}
		}

		make_unit_vector(grad);
		return grad;
	}

	/*
		Train the network using td and gradient descent until the cost()
		function is less than the endpoint_cost or exceeds max_iterations.
	*/
	void train(const float &endpoint_cost, const unsigned &max_iterations)
	{
		unsigned iterations = 0;
		while (cost() > endpoint_cost && iterations < max_iterations)
		{
			//printf("Cost: %*1.8f\n", 15, cost());

			vector<float> grad_w = unit_grad_cost_w();
			vector<float> grad_b = unit_grad_cost_b();
			
			auto iter_w = grad_w.begin();
			auto iter_b = grad_b.begin();

			// Adjust each weight and bias according to the gradient.
			for(unsigned i = 0; i < weights.size(); i++)
			{
				for(unsigned j = 0; j < weights[i].size(); j++)
				{
					for(unsigned k = 0; k < weights[i][j].size(); k++)
					{
						weights[i][j][k] -= *iter_w*learning_rate;
						iter_w++;
					}
					biases[i][j] -= *iter_b*step;
					iter_b++;
				}
			}

			iterations++;
		}

		if(iterations == max_iterations)
			printf("Reached max iterations: %d.\n", max_iterations);
		else
			printf("Converged Successfully!\n");
	}

	// Standard vector magnitude. Returns |v|.
	float magnitude_vec(const vector<float> &u)
	{
		float sum = 0;

		for(const auto &n : u)
			sum += n*n;

		return sqrtf(sum);
	}

	// Converts passed vector to a unit vector.
	void make_unit_vector(vector<float> &u)
	{
		float mag = magnitude_vec(u);

		for(auto &n : u)
			n/=mag;
	}

	// Standard vector dot product. Returns u*v.
	float dot_vec(const vector<float> &u, const vector<float> &v)
	{
		assert(u.size() == v.size());

		float sum = 0;

		for(int i = 0; i < u.size(); i++)
			sum += u[i]*v[i];
		
		return sum;
	}

	// Standard vector subtraction. Returns u-v.
	vector<float> subtract_vec(vector<float> u, vector<float> v)
	{
		assert(u.size() == v.size());

		vector<float> ans;

		for(int i = 0; i < u.size(); i++)
			ans.push_back(u[i] - v[i]);

		return ans;
	}

	// Debugging
	void print()
	{
		unsigned i = 0;
		for(const auto layer : weights)
		{
			printf("---------- %d ----------\n", i);
			unsigned j = 0;
			for(const auto neuron : layer)
			{
				printf("Bias:%*1.8f\n", 18, biases.at(i).at(j));
				int i = 0;
				for(const auto weight : neuron)
				{
					if(!i++)
						printf("Weights:%*1.8f\n", 15, weight);
					else
					printf("\t%*1.8f\n", 15, weight);
				}
				printf("\n");
				j++;
			}
			printf("---------- %d ----------\n\n", i);
			i++;
		}
	}
};
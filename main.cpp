#include <iostream>
#include "network.hpp"
#include "data_types.hpp"


void print_vec(const vector<float> &v)
{
	int i = 0;
	for(const auto &n : v)
	{
		if(!i++)
			printf("[%*1.15f", 19, n);
		else
			printf("\n%*1.15f", 20, n);
	}

	printf("  ]\n");
}

/*
	Generate training data.

	inputs contains 5 random integers and outputs contain 
	either {1,0} if sum of inputs is even, and {0,1} if sum is odd.
*/
Training_Data generate_training_data(int num_inputs, int samples)
{
	srand(0);

	Training_Data  td;
	std::vector<float> in;

	int rand_num;

	for(int i = 0; i < samples; i++)
	{
		int sum = 0;

		for(int j = 0; j < num_inputs; j++)
		{
			rand_num = rand()%5;
			in.push_back(rand_num);
			sum += rand_num;
		}

		td.inputs.push_back(in);

		// 1,0 on the output means even, 0,1 odd.
		if(sum%2 == 0)
			td.outputs.push_back({ 1, 0 });
		else
			td.outputs.push_back({ 0, 1 });

		in.clear();
	}

	return td;
}

void test_net(const Training_Data &td, Network &nn)
{
	vector<float> outs;
	unsigned correct = 0;
	for(unsigned i = 0; i < td.inputs.size(); i++)
	{
		outs = nn.output(td.inputs[i], nn.layers-1);

		// Even
		if(outs[0] > outs[1] && td.outputs[i][0] == 1)
			correct++;
		else if(outs[0] < outs[1] && td.outputs[i][1] == 1)
			correct++;
	}

	printf("When testing with %d inputs, net is correct %d times.\n",
			td.inputs.size(), correct);
}

int main()
{
	Network nn = Network({ 5, 10, 2 }, 0.1);

	printf("\n\n\n");
	printf("Done initializing net!\n");

	nn.td = generate_training_data(5, 1000);


	nn.print();

	printf("Untrained:\n");
	test_net(nn.td, nn);
	printf("\nTraining\n");

	nn.train(0.4, 1000);
	nn.learning_rate = 0.1;
	nn.train(0.3, 1000);
	nn.learning_rate = 0.05;
	nn.train(0.2, 1000);
	nn.learning_rate = 0.01;
	nn.train(0.15, 1000);
	//nn.learning_rate = 0.005;

	printf("Training complete!\n");

	test_net(nn.td, nn);
	printf("\n\n\n");

	return 0;
}

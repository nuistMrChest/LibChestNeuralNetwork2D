#include<iostream>
#include"nn/tensor_3d.hpp"
#include"nn/layer.hpp"

using namespace LibCN;

int main(){
	CNNLayer<double> layer(
		3, // i_c
		3, // i_h
		3, // i_l
		1, // o_c
		2, // o_h
		2, // o_l
		1, // stride
		0  // padding
	);

	layer.activation=[](const Tensor3d<double>&x){
		return x;
	};

	layer.activation_d=[](const Tensor3d<double>&x){
		Tensor3d<double>res(x.c,x.h,x.l);
		for(size_t i=0;i<res.v.size();i++)res.v[i]=1.0;
		return res;
	};

	Tensor3d<double> input{
		{
			{1,2,0},
			{0,1,3},
			{2,1,1}
		},
		{
			{2,1,1},
			{1,0,2},
			{3,1,0}
		},
		{
			{0,1,2},
			{2,1,0},
			{1,3,1}
		}
	};

	layer.kernal.resize(1);
	layer.kernal[0]=Tensor3d<double>{
		{
			{1,0},
			{0,-1}
		},
		{
			{0,1},
			{1,0}
		},
		{
			{1,-1},
			{0,1}
		}
	};

	layer.b.resize(1);
	layer.b[0]=1.0;

	std::cout<<"input =\n"<<input<<"\n\n";
	std::cout<<"kernel[0] before =\n"<<layer.kernal[0]<<"\n\n";
	std::cout<<"b before = "<<layer.b[0]<<"\n\n";

	auto out=layer.forward(input);

	std::cout<<"forward output =\n"<<out<<"\n\n";
	std::cout<<"z =\n"<<layer.z<<"\n\n";

	Tensor3d<double> dl_da{
		{
			{1,2},
			{3,4}
		}
	};

	std::cout<<"dl_da =\n"<<dl_da<<"\n\n";

	double step=0.1;
	auto dl_prev=layer.backward(dl_da,step);

	std::cout<<"dl_prev =\n"<<dl_prev<<"\n\n";
	std::cout<<"kernel[0] after =\n"<<layer.kernal[0]<<"\n\n";
	std::cout<<"b after = "<<layer.b[0]<<"\n";

	return 0;
}

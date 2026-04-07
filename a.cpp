#include "lib_chest_nn.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace LibCN;

static Tensor3d<double> deflatten1(const Matrix<double>&a,size_t h,size_t l){
	Tensor3d<double>res(1,h,l);
	res.v=a.v;
	return res;
}

int main(){
	using T=double;

	std::vector<Tensor3d<T>>xs={
		{{{0,1,0,1},{0,1,0,1},{0,1,0,1},{0,1,0,1}}},
		{{{1,0,1,0},{1,0,1,0},{1,0,1,0},{1,0,1,0}}},
		{{{0,0,0,0},{1,1,1,1},{0,0,0,0},{1,1,1,1}}},
		{{{1,1,1,1},{0,0,0,0},{1,1,1,1},{0,0,0,0}}}
	};

	std::vector<Matrix<T>>ys={
		{{1}},
		{{1}},
		{{0}},
		{{0}}
	};

	CNNLayer<T>conv(1,4,4,1,2,2,1,0);
	conv.activation=
		static_cast<Tensor3d<T>(*)(const Tensor3d<T>&)>(&Activations::tanh<T>);
	conv.activation_d=
		static_cast<Tensor3d<T>(*)(const Tensor3d<T>&)>(&Activations::tanh_d<T>);
	conv.init(1,1,3,3,T(-1),T(1));

	MLP<T>mlp(2,4,1,T(0.1));
	mlp.layers[0]=MLPLayer<T>(4,4);
	mlp.layers[1]=MLPLayer<T>(4,1);

	mlp.layers[0].activation=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::tanh<T>);
	mlp.layers[0].activation_d=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::tanh_d<T>);

	mlp.layers[1].activation=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::sigmoid<T>);
	mlp.layers[1].activation_d=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::sigmoid_d<T>);

	mlp.setLoss(Losses::MSE<T>,Losses::MSE_d<T>);
	mlp.init(T(-1),T(1));

	T conv_step=0.1;

	for(size_t epoch=0;epoch<5000;epoch++){
		T loss_sum=0;
		for(size_t i=0;i<xs.size();i++){
			Tensor3d<T>conv_out=conv.forward(xs[i]);

			Matrix<T>dl_da_flat;
			loss_sum+=mlp.train(conv_out.flatten(),ys[i],dl_da_flat);

			Tensor3d<T>dl_da_conv=deflatten1(dl_da_flat,2,2);
			conv.backward(dl_da_conv,conv_step);
		}

		if(epoch%500==0){
			std::cout<<"epoch="<<epoch
				<<" avg_loss="<<(loss_sum/xs.size())<<"\n";
		}
	}

	std::cout<<std::fixed<<std::setprecision(6);
	for(size_t i=0;i<xs.size();i++){
		Tensor3d<T>conv_out=conv.forward(xs[i]);
		Matrix<T>pred=mlp.use(conv_out.flatten());
		std::cout
			<<"sample "<<i
			<<" pred="<<pred(0,0)
			<<" target="<<ys[i](0,0)
			<<"\n";
	}

	return 0;
}

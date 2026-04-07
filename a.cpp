#include "lib_chest_nn.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace LibCN;

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

	CNN<T>cnn;
	cnn.step=T(0.1);
	cnn.i_c=1;
	cnn.i_h=4;
	cnn.i_l=4;
	cnn.layers.resize(1);

	cnn.layers[0]=CNNLayer<T>(1,4,4,1,2,2,1,0);
	cnn.layers[0].activation=
		static_cast<Tensor3d<T>(*)(const Tensor3d<T>&)>(&Activations::tanh<T>);
	cnn.layers[0].activation_d=
		static_cast<Tensor3d<T>(*)(const Tensor3d<T>&)>(&Activations::tanh_d<T>);
	cnn.layers[0].init(1,1,3,3,T(-1),T(1));

	cnn.o_c=1;
	cnn.o_h=2;
	cnn.o_l=2;

	cnn.mlp=MLP<T>(2,4,1,T(0.1));
	cnn.mlp.layers[0]=MLPLayer<T>(4,4);
	cnn.mlp.layers[1]=MLPLayer<T>(4,1);

	cnn.mlp.layers[0].activation=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::tanh<T>);
	cnn.mlp.layers[0].activation_d=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::tanh_d<T>);

	cnn.mlp.layers[1].activation=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::sigmoid<T>);
	cnn.mlp.layers[1].activation_d=
		static_cast<Matrix<T>(*)(const Matrix<T>&)>(&Activations::sigmoid_d<T>);

	cnn.mlp.setLoss(Losses::MSE<T>,Losses::MSE_d<T>);
	cnn.mlp.init(T(-1),T(1));

	for(size_t epoch=0;epoch<5000;epoch++){
		T loss_sum=T(0);
		for(size_t i=0;i<xs.size();i++){
			loss_sum+=cnn.train(xs[i],ys[i]);
		}

		if(epoch%500==0){
			std::cout
				<<"epoch="<<epoch
				<<" avg_loss="<<(loss_sum/xs.size())
				<<"\n";
		}
	}

	std::cout<<std::fixed<<std::setprecision(6);
	for(size_t i=0;i<xs.size();i++){
		Matrix<T>pred=cnn.mlp.use(cnn.layers[0].forward(xs[i]).flatten());
		std::cout
			<<"sample "<<i
			<<" pred="<<pred(0,0)
			<<" target="<<ys[i](0,0)
			<<"\n";
	}

	return 0;
}

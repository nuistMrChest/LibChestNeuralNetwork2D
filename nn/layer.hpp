#ifndef LAYER_HPP
#define LAYER_HPP

#include"matrix.hpp"
#include"tensor_3d.hpp"
#include<functional>
#include<random>

namespace LibCN{
	template<Element T>struct MLPLayer{
		std::function<Matrix<T>(const Matrix<T>&)>activation;
		std::function<Matrix<T>(const Matrix<T>&)>activation_d;
		size_t in_size;
		size_t out_size;
		Matrix<T>W;
		Matrix<T>b;
		Matrix<T>last_input;
		Matrix<T>z;
		bool sm;

		MLPLayer(){
			in_size=0;
			out_size=0;
			W=Matrix<T>();
			b=Matrix<T>();
			last_input=Matrix<T>();
			z=Matrix<T>();
			sm=false;
		}

		MLPLayer(size_t i,size_t o){
			in_size=i;
			out_size=o;
			W.resize(o,i);
			b.resize(o,1);
			last_input.resize(i,1);
			z.resize(o,1);
			sm=false;
		}

		Matrix<T>forward(const Matrix<T>&input){
			Matrix<T>res(out_size,1);
			last_input=input;
			if(input.h==in_size&&input.l==1)z=((W*input)+b);
			res=activation(z);
			return res;
		}

		Matrix<T>backward(const Matrix<T>&dl_da,const T&step){
			Matrix<T>res;
			Matrix<T>dl_dz=dl_da.hadamard(activation_d(z));
			res=W.transpose()*dl_dz;
			W-=step*(dl_dz*last_input.transpose());
			b-=step*dl_dz;
			return res;
		}

		Matrix<T>backward_dz(const Matrix<T>&dl_dz,const T&step){
			Matrix<T>res=W.transpose()*dl_dz;
			W-=step*(dl_dz*last_input.transpose());
			b-=step*dl_dz;
			return res;
		}
	
		void init(T low=T(-1),T high=T(1)){
			static std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<T>dist(low,high);
			for(size_t i=0;i<out_size;++i){
				for(size_t j=0;j<in_size;++j){
					W(i,j)=dist(rng);
				}
				b(i,0)=dist(rng);
			}
		}
	};

	template<Element T>struct CNNLayer{
		std::function<Tensor3d<T>(const Tensor3d<T>&)>activation;
		std::function<Tensor3d<T>(const Tensor3d<T>&)>activation_d;
		Tensor4d<T>kernal;
		size_t i_c,i_h,i_l;
		size_t o_c,o_h,o_l;
		size_t stride,padding;
		Tensor3d<T>lase_input;
		Tensor3d<T>b;

		CNNLayer(){
			i_c=0;
			i_h=0;
			i_l=0;
			o_c=0;
			o_h=0;
			o_l=0;
		}

		CNNLayer(
			size_t i_c,
			size_t i_h,
			size_t i_l,
			size_t o_c,
			size_t o_h,
			size_t o_l,
			size_t s,
			size_t p
		){
			this->i_c=i_c;
			this->i_h=i_h;
			this->i_l=i_l;
			this->o_c=o_c;
			this->o_h=o_h;
			this->o_l=o_l;
			this->stride=s;
			this->padding=p;
			b.resize(o_c,o_h,o_l);
		}

		void init(size_t c_o,size_t c_i,size_t h,size_t l,T low=T(-1),T high=T(1)){
			static std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<T>dist(low,high);
			kernal.resize(c_o);
			for(size_t i=0;i<c_o;i++){
				kernal[i].resize(c_i,h,l);
				for(size_t j=0;j<kernal[i].v.size();j++)kernal[i].v[j]=dist(rng);
			}
			for(size_t i=0;i<b.v.size();i++)b.v[i]=dist(rng);
		}

		Tensor3d<T>forward(const Tensor3d<T>&input){
			Tensor3d<T>res;
			if(input.c==i_c&&input.h==i_h&&input.l==i_l){
				res=input.convolution(kernal,stride,padding)+b;
				res=activation(res);
			}
			return res;
		}
	};
}

#endif

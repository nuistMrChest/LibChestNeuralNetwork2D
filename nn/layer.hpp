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
		std::vector<T>b;
		Tensor3d<T>z;
		Tensor3d<T>last_input;

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
			b.resize(o_c);
		}

		void init(size_t c_o,size_t c_i,size_t h,size_t l,T low=T(-1),T high=T(1)){
			static std::mt19937 rng(std::random_device{}());
			std::uniform_real_distribution<T>dist(low,high);
			kernal.resize(c_o);
			for(size_t i=0;i<c_o;i++){
				kernal[i].resize(c_i,h,l);
				for(size_t j=0;j<kernal[i].v.size();j++)kernal[i].v[j]=dist(rng);
			}
			for(size_t i=0;i<b.size();i++)b[i]=dist(rng);
		}

		Tensor3d<T>forward(const Tensor3d<T>&input){
			Tensor3d<T>res;
			last_input=input;
			if(input.c==i_c&&input.h==i_h&&input.l==i_l){
				z=input.convolution(kernal,stride,padding);
				for(size_t i=0;i<o_c;i++)
					for(size_t x=0;x<o_h;x++)
						for(size_t y=0;y<o_l;y++)
							z(i,x,y)+=b[i];
				res=activation(z);
			}
			return res;
		}

		Tensor3d<T>backward(const Tensor3d<T>&dl_da,const T&step){
			Tensor3d<T>res(i_c,i_h,i_l);

			for(size_t i=0;i<res.v.size();i++)res.v[i]=T(0);
			Tensor3d<T>dl_dz=dl_da.hadamard(activation_d(z));

			for(size_t j=0;j<i_c;j++)
				for(size_t a=0;a<i_h;a++)
					for(size_t b=0;b<i_l;b++){
						res(j,a,b)=T(0);
						for(size_t i=0;i<o_c;i++)
							for(size_t u=0;u<o_h;u++)
								for(size_t v=0;v<o_l;v++)
									if(
										!(
											(long long)a-(long long)(stride*u)+(long long)padding<0||
											(long long)b-(long long)(stride*v)+(long long)padding<0||
											(long long)a-(long long)(stride*u)+(long long)padding>=(long long)kernal[0].h||
											(long long)b-(long long)(stride*v)+(long long)padding>=(long long)kernal[0].l
										)
									)
									res(j,a,b)+=
										dl_dz(i,u,v)*
										kernal[i](
											j,
											(long long)a-(long long)(u*stride)+(long long)padding,
											(long long)b-(long long)(v*stride)+(long long)padding
										);
					}

			for(size_t i=0;i<o_c;i++)
				for(size_t j=0;j<i_c;j++)
					for(size_t x=0;x<kernal[0].h;x++)
						for(size_t y=0;y<kernal[0].l;y++){
							T grad_k=T(0);
							for(size_t u=0;u<o_h;u++)
								for(size_t v=0;v<o_l;v++){
									if(
										!(
											(long long)(u*stride)+(long long)x-(long long)padding<0||
											(long long)(v*stride)+(long long)y-(long long)padding<0||
											(long long)(u*stride)+(long long)x-(long long)padding>=(long long)i_h||
											(long long)(v*stride)+(long long)y-(long long)padding>=(long long)i_l
										)
									)
									grad_k+=
										dl_dz(i,u,v)*
										last_input(
											j,
											(long long)(u*stride)+(long long)x-(long long)padding,
											(long long)(v*stride)+(long long)y-(long long)padding
										);
								}
							kernal[i](j,x,y)-=step*grad_k;
						}
			std::vector<T>grad_b(o_c,T(0));

			for(size_t i=0;i<o_c;i++)
				for(size_t u=0;u<o_h;u++)
					for(size_t v=0;v<o_l;v++)
						grad_b[i]+=dl_dz(i,u,v);

			for(size_t i=0;i<o_c;i++)b[i]-=step*grad_b[i];

			return res;
		}
	};
}

#endif

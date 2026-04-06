#include"nn/tensor_3d.hpp"
#include<iostream>

using namespace std;
using namespace LibCN;

int main(){
	Tensor3d<int> x={
		{
			{1,2,3},
			{4,5,6},
			{7,8,9},
			{10,11,12}
		},
		{
			{10,20,30},
			{40,50,60},
			{70,80,90},
			{100,110,120}
		}
	};
		Tensor4d<int> w={
		// kernel 0
		{
			{
				{1,2},
				{0,1}
			},
			{
				{1,0},
				{0,1}
			}
		},

		// kernel 1
		{
			{
				{0,1},
				{1,0}
			},
			{
				{1,1},
				{1,0}
			}
		}
	};
	cout<<x.convolution(w,2,1)<<endl;
}

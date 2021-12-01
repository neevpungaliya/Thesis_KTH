#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
 
int main()
{
    int i_dim;
    int f_dim;
    int pad;
    int stride;
    int o_dim;
    int mid_f_dim;
    int ker;
//--------------------------------------------------------
     // CONVOLUTION LAYER BEGINS HERE
 
    srand((unsigned int)time(NULL));
    //inputs taken
    printf("Enter the input dimensions: \n");
    scanf("%d", &i_dim);
 
    //printf("Enter the number of kernels: \n");
    //scanf("%d", &ker);
 
    printf("Enter the filter dimensions: \n");
    scanf("%d", &f_dim);
 
    printf("Enter amount of padding required: \n");
    scanf("%d", &pad);
 
    printf("Enter stride value: \n");
    scanf("%d", &stride);
 
    //Output layer and input padded layer dimension calculations
    o_dim = floor(((i_dim - f_dim + 2*pad )/stride))+1;
    int i_dim_pad=(2*pad)+i_dim;
printf("\n_________________________________________________________________________________________________________________\n");
    //array declarations
    int arr_i_dim[i_dim][i_dim];
    float arr_f_dim[f_dim][f_dim];
    float arr_o_dim[o_dim][o_dim];
    int arr_i_dim_pad[i_dim_pad][i_dim_pad];
 
    //random 2D data added to input layer
    printf("Input matrix without padding - \n");
    for(int i=0;i<i_dim;i++){
        for(int j=0;j<i_dim;j++){
            arr_i_dim[i][j]=(rand()%10);
            printf("%d \t", arr_i_dim[i][j]);
        }
        printf("\n");
    }
    printf("\n \n");
 
    //random 2D data added to filter
    printf("Filter matrix - \n");
    float a=1.0;
    for(int i=0;i<f_dim;i++){
        for(int j=0;j<f_dim;j++){
            arr_f_dim[i][j]=((float)rand()/(float)(RAND_MAX)) * a;
            printf("%f \t", arr_f_dim[i][j]);
        }
        printf("\n");
    }
    printf("\n \n");
 
    //padding added [new input matrix is stored in arr_i_dim_pad]
    if(pad!=0){
        printf("Input matrix with padding - \n");
        for(int i=0;i<(i_dim_pad);i++){
            for(int j=0;j<(i_dim_pad);j++){
                if(j<pad || j>=(i_dim_pad-pad) || i<pad || i>=(i_dim_pad-pad)){
                    arr_i_dim_pad[i][j]=0;
                    printf("%d \t", arr_i_dim_pad[i][j]);
                }
                else{
                    arr_i_dim_pad[i][j]=arr_i_dim[i-pad][j-pad];
                    printf("%d \t", arr_i_dim_pad[i][j]);
                }
            }
            printf("\n");
        }
    }
    else{
        for(int i=0;i<(i_dim_pad);i++){
            for(int j=0;j<(i_dim_pad);j++){
                arr_i_dim_pad[i][j]=arr_i_dim[i][j];
            }
        }
    }
    printf("Output dimensions: %d \n", o_dim);
 
    //calculation of output values, for iteration on output matrix
    printf("Output matrix - \n");
    float sum=0.0;
    for(int i=0;i<=(i_dim_pad-f_dim);i=i+stride){              //rows of input padded matrix
        for(int j=0;j<=(i_dim_pad-f_dim);j=j+stride){          //columns of input padded matrix
            sum=0.0;
            for(int p=0;p<f_dim;p++){
                for(int q=0;q<f_dim;q++){
                    sum += arr_f_dim[p][q] * arr_i_dim_pad[i+p][j+q];
                }
            }
        arr_o_dim[i][j]=sum;
        printf("%f \t", arr_o_dim[i][j]);
        }
        printf("\n \n");
    }
    printf("\n_________________________________________________________________________________________________________________\n");
//--------------------------------------------------------------
    // FULLY CONNECTED LAYER BEGINS HERE
    int flatten_size=o_dim*o_dim;
    printf("\nFlatten size: %d\n", flatten_size);
    float flatten_out[flatten_size];
    printf("Flattened matrix - \n");
    for(int i=0;i<o_dim;i++){         //flatten
        for(int j=0;j<o_dim;j++){
            flatten_out[(i*o_dim)+j]=arr_o_dim[i][j];
            printf("%f \n", flatten_out[(i*o_dim)+j]);
        }
    }
 
//now consider 1 hidden layer of size flatten_out*2/3 + number of output nodes
    int hidden_dim = floor((flatten_size*2)/3)+2;
    printf("\nHidden size: %d\n", hidden_dim);
    float arr_hidden_dim[hidden_dim];
    //float sumh[hidden_dim+1];
    //float weight[flatten_size+1][hidden_dim+1];
 
    float bias[hidden_dim];
    float sumh[hidden_dim];
    float weight[flatten_size][hidden_dim];
 
    printf("\nWeight matrix for hidden layer of size %d * %d- \n", flatten_size,hidden_dim);
    float maxi=1.0;
    float min=-1.0;
 
    //weights are randomized here between -1 and 1
    for(int i=0;i<flatten_size;i++){      //randomize weights
        for(int j=0;j<hidden_dim;j++){
            weight[i][j]=(((float)rand()/(float)(RAND_MAX)) * (maxi-min))+min;
            printf("%f \t", weight[i][j]);
        }
        printf("\n");
    }
    printf("\n");
 
    //biases are randomized here
    printf("Biases are as follows: \n");
    for(int i=0;i<hidden_dim;i++){
        bias[i]=((float)rand()/(float)(RAND_MAX)) * 1.0;
        printf("%f \n", bias[i]);
    }
 
    printf("\n\nHidden Layer - \n");
    printf("\nWithout activation | With sigmoid\n");
 
    //calculation forward propagation on 1st hidden layer
    for(int j=0;j<hidden_dim;j++){
        sumh[j]=bias[j];
        for(int i=0;i<flatten_size;i++){
            sumh[j]+=weight[i][j]*flatten_out[i];
        }
        arr_hidden_dim[j]=1.0/(1.0+exp(-sumh[j]));
        printf("%f \t\t%f\n", sumh[j],arr_hidden_dim[j]);
    }
printf("\n_________________________________________________________________________________________________________________\n");
    //Output layer after 1 hidden layer, number of output nodes=2
    int output_dim=2;
    float arr_output_dim[output_dim];
    float biaso[output_dim];
    float sumo[output_dim];
    float weighto[hidden_dim][output_dim];
    printf("\nOutput nodes: %d\n", output_dim);
 
    printf("\nWeight matrix for output layer of size %d * %d- \n", hidden_dim,output_dim);
    //float maxi=1.0;
    //float min=-1.0;
    for(int i=0;i<hidden_dim;i++){      //randomize weights
        for(int j=0;j<output_dim;j++){
            weighto[i][j]=(((float)rand()/(float)(RAND_MAX)) * (maxi-min))+min;
            printf("%f \t", weight[i][j]);
        }
        printf("\n");
    }
    printf("\n");
 
    printf("Biases are as follows: \n");
    for(int i=0;i<output_dim;i++){
        biaso[i]=((float)rand()/(float)(RAND_MAX)) * 1.0;
        printf("%f \n", bias[i]);
    }
 
    printf("\n\nOutput Layer - \n");
    printf("\nWithout activation | With sigmoid\n");
 
    //calculation forward propagation on 1st hidden layer
    for(int j=0;j<output_dim;j++){
        sumo[j]=biaso[j];
        for(int i=0;i<hidden_dim;i++){
            sumo[j]+=weighto[i][j]*arr_hidden_dim[i];
        }
        arr_output_dim[j]=1.0/(1.0+exp(-sumo[j]));
        printf("%f \t\t%f\n", sumo[j],arr_output_dim[j]);
    }
 
}

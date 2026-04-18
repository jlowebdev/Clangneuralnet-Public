#ifndef STRUCTFILE_H
#define STRUCTFILE_H
#include <stdio.h>
//Global Enums for fixed math
enum {length = 26};
//Perceptrons
// For a single non-connected perceptron just call it once otherwise just chain inputs to outputs
struct perc{
    float inputs[2];
    float weights[2];
    float max[2]; // Will be max x and y values 
    float min[2]; //Minimum x and y values
    int outputs[2]; // Will represent the binary choice between two classes for that perceptron, when classifying it will use the value here as the output value
    float bias;
    float output;
    int pred;
    int last; // Identifies as the last in the chain
    int clone; // Will identify when the network must forcibly get at least one wrong 
    struct perc * prev;
    struct perc * next;

};

struct perc_n{

    int n_dims;
    float *weights;
    float *min;
    float *max;
    float bias;
    float output;
    int pred;
    int last;
    struct perc_n * prev;
    struct perc_n * next;
};

struct points{
    float coords[1][2];
};

struct set{
    int firstdim;
    int seconddim;
    int outputs[2]; //options for potential output -1 will be index 0 and output 1 will be the equivalent of 1;
    float ** coords;
    int * labels;
    
    
    

};
struct set_n{
    int n_points;
    int n_dims;
    int dim;  //This will indicate the dimension of the points in the array and will pair index wise with the other dimensions for easier access.
    float * coords;
    int *labels;
};

//Modern Digital Neuron (Node)
struct node{
    float weight[length];
    float value;
    float delta;
    int active[length];
    struct node * next;    
};
struct outnode{
    float prediction;

};
//File and Data Handling
struct file_info{
    size_t filesz;
    char * carry;
};
struct token{
    int idx;
    int nextlidx;
    float val;


};


#endif
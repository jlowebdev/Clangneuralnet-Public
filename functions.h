#ifndef FUNCTIONSFILE_H
#define FUNCTIONSFILE_H
#include "structs.h"

//Perceptron
void smart_weight_init(struct perc * ada, struct set * temp);
void perc_random_init(struct perc * perc);
void calc_perceptron_output(struct perc * perc);
void perc_activation(struct perc * perc);
void train_perc(struct perc * perc, float target);
        // Perceptron Tools
int float_equal(float a, float b);
struct set * copy_set(struct set * temp);
struct set * rmv_point (struct set * temp, float x_max, float x_min);
struct set * ret_tempf(struct perc * ada, struct set * temp);
struct set * rename_set(struct set * temp);
void free_set(struct set * temp);
        //Perceptron Experimental
float covar_activ (int firstdim, int seconddim, float data[firstdim][seconddim]);
float average_slope_group (int firstdim, int seconddim, float data[firstdim][seconddim]);
struct perc * create_perc ();
struct set * create_subset (int firstdim, int seconddim, float data[firstdim][seconddim], int labels[firstdim]);
void find_perc_bounds(struct perc * ada, struct set * set);
int perceptron_trav(struct perc * ada, float inputx, float inputy, float threshold, int depth);

void train_perc_stagsp (struct perc * ada, struct set * hold_subset);
struct perc * veri_perc_chain2(struct perc * ada, int firstdim, int seconddim, float data[firstdim][seconddim], int labels[firstdim], int depth);

//ML Modern
void delta_calc(struct node * alayer, struct node * blayer, int alen, int blen);
void final_layer_delta_calc(struct node * alayer, struct node * blayer, int alen, int blen);
void sigmoid(struct node * alayer, int alen, struct node * goals);
void clear_nodes(struct node * alayer, int alen);
void define_goals(struct node * alayer, struct node * blayer, int alen, int c_target);
void back_prop(struct node * alayer, struct node * blayer, int alen, int blen, int cmp_f);
float neural_net(float * inputs, float * weights);
float error(float pred, float true_w);
float delta(float pred, float true_g);
struct node * nodes_gen(int len, int next_layer_len);
        // Modern ML Experimental
void hidden_math(struct node * alayer, struct node * blayer, int alen, int blen, int isout);
void lth_hidden_math(struct node * alayer, struct node * blayer, int alen, int blen, int isout);
void pruning(struct node * alayer, struct node * blayer, int alen, int blen, float minabsval);
int density(struct node * alayer, int alen, int blen);
        //Associated Tools
struct token * token_gen(char * arry, int len_of_arry);
float weighted_sum(float * a, float * b, int len);

#endif
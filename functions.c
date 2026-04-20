#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "structs.h"


//Perceptron Section
// Will have perceptron library components for training perceptrons along with some testing
// that may be relevant to modern day ML as well

//Single Perceptron and Weight Stuff primarily binary classifier
void perc_random_init(struct perc * perc){
    float limit = sqrt(6.0f / (1 + 1));
    for (int i = 0; i<2; i++){
        perc->weights[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;

    }
    perc->bias = 0.0f;
}
void smart_weight_init(struct perc * ada, struct set * temp){
    //Find center points of both datasets
    float mean_pos_x = 0, mean_pos_y = 0;
    float mean_neg_x = 0, mean_neg_y = 0;
    int pos_count = 0, neg_count = 0;

    for (int i = 0; i < temp->firstdim; i++){
        if (temp->labels[i] == 1){
            mean_pos_x += temp->coords[i][0];
            mean_pos_y += temp->coords[i][1];
            pos_count++;
        } else {
            mean_neg_x += temp->coords[i][0];
            mean_neg_y += temp->coords[i][1];
            neg_count++;
        }
    }

    mean_pos_x /= pos_count; 
    mean_pos_y /= pos_count;
    mean_neg_x /= neg_count;
    mean_neg_y /= neg_count;

    
    float w_x = mean_pos_x - mean_neg_x;
    float w_y = mean_pos_y - mean_neg_y;

    float mag = sqrtf(w_x*w_x + w_y*w_y);
    ada->weights[0] = w_x / mag;
    ada->weights[1] = w_y / mag;

    // Bias at midpoint between centroids
    float mid_x = (mean_pos_x + mean_neg_x) / 2.0f;
    float mid_y = (mean_pos_y + mean_neg_y) / 2.0f;
    ada->bias = -(ada->weights[0] * mid_x + ada->weights[1] * mid_y);
    if (temp->firstdim == 1){
        perc_random_init(ada);
    }
}

void calc_perceptron_output(struct perc * perc){
    perc->output = perc->inputs[1]*perc->weights[1] + perc->inputs[0]*perc->weights[0] + perc->bias;

}
void perc_activation(struct perc * perc){
    // perc->pred = roundf(perc->output);
    if (perc->output > 0){
        perc->pred = 1;
    }
    else{
        perc->pred = -1;
    }
}
void train_perc(struct perc * perc, float target){
    for (int i = 0; i < 2; i++){
        perc->weights[i] += 0.001f * (target - perc->pred) * perc->inputs[i];
    }
    perc->bias += 0.1f * (target - perc->pred);



}
// Covariance Preprocessing for perceptron testing
float covar_activ (int firstdim, int seconddim, float data[firstdim][seconddim]){
    float sumx = 0.0f, sumy = 0.0f;
    float covar_for_group = 0.0f;
    for(int i = 0; i< firstdim; i++){
        sumx += data[i][0];
        sumy += data[i][1];

    }
    for (int i = 0; i < firstdim; i++){

        covar_for_group += (data[i][0] - (sumx/firstdim))*(data[i][1] - (sumy/firstdim));

    }
    covar_for_group = covar_for_group/firstdim;
    return covar_for_group;
}

float average_slope_group (int firstdim, int seconddim, float data[firstdim][seconddim]){
    float slope = 0.0f;
    for(int i = 0; i< firstdim; i++){
        slope += (data[i+1][1] - data[i][1]) /
                 (data[i+1][0] - data[i][0]);
        if (i == firstdim-1){
            (data[i+1][1] - data[i-1][1]) /
            (data[i+1][0] - data[i-1][0]);
        }
    }
    slope /= firstdim;
    return slope;

}

//For the idea of dividing parts where the perceptron struggles into further parts and training perceptrons on those
// Will use a confidence cascade to make the prediction on new data however training works by finding the most you can divide the data with a single line,
// then taking the points it fails on and doing the best it can with a single line there until it gets to two points at which point if it is unable to divide
// with a straight line it will use a non-linear activation but only for the extreme case
// Will also have to classify based on coordinates, new perceptrons will have a coordinate range that they will cover in the perceptron data structure

struct perc * create_perc (){ //Creates a new two weight perceptron and returns the address
    struct perc * ada = calloc(1,sizeof(struct perc));
    perc_random_init(ada);
    return ada;
};

struct set * create_subset (int firstdim, int seconddim, float data[firstdim][seconddim], int labels[firstdim]){ // takes the uncertain points and creates a new subset where a new perceptron will try to classify the best it can
// Must have the associated training labels to go along with it 
// Just allocates the set from data to the set data type
    struct set * set = calloc(1, sizeof(struct set));
    set->firstdim = firstdim;
    set->coords = malloc(set->firstdim * sizeof(float *));
    set->labels = calloc(set->firstdim, sizeof(int));
    for(int i = 0; i<firstdim; i++){
        set->coords[i] = calloc(2, sizeof(float));
        set->coords[i][0] = data[i][0];
        set->coords[i][1] = data[i][1];
        set->labels[i] = labels[i];
    }
    return set;
};

void find_perc_bounds(struct perc * ada, struct set * set){
// For any subset find the max and min x and y values that its applicable for then set the proper bounds in the perc data struct
    float miny = FLT_MAX;
    float minx = FLT_MAX;
    float maxy = -FLT_MAX;
    float maxx = -FLT_MAX;
    float pad = 0.01f;

    for(int i = 0; i<set->firstdim; i++){

        if(miny > set->coords[i][1]){
            miny = set->coords[i][1];
        }
        if(maxy < set->coords[i][1]){
            maxy = set->coords[i][1];
        }
        if(minx > set->coords[i][0]){
            minx = set->coords[i][0];
        }
        if(maxx < set->coords[i][0]){
            maxx = set->coords[i][0];
        }

    }
    ada->min[0] = minx - 0.001f;
    ada->min[1] = miny - 0.001f;
    ada->max[0] = maxx + 0.001f;
    ada->max[1] = maxy + 0.001f;


}



int perceptron_trav(struct perc * ada, float inputx, float inputy, float threshold, int depth){

    ada->inputs[0] = inputx;
    ada->inputs[1] = inputy;


    calc_perceptron_output(ada);
    perc_activation(ada);
    float eps = 1e-4f;
    int in_bounds =
    (inputx >= ada->min[0] - eps && inputx <= ada->max[0] + eps) &&
    (inputy >= ada->min[1] - eps && inputy <= ada->max[1] + eps);

    float confidence = fabs(ada->output);

    if ((!in_bounds) && ada != NULL){
        // printf("Confidence before routing:%0.5f\n", confidence);


        return perceptron_trav(ada->prev, inputx, inputy, threshold, depth+1);


    }
    printf("Predicted as: %d, with %f confidence at depth %d.\n", ada->pred, confidence, depth);

    return ada->pred;
}
int float_equal(float a, float b) {
    return fabs(a - b) <= 0.001f;
}
struct set * copy_set(struct set * temp){
    //just a copy function


    struct set * new = malloc(sizeof(struct set));
    new->firstdim = temp->firstdim;
    new->coords = malloc(new->firstdim * sizeof(float *));
    for (int i = 0; i<new->firstdim; i++){
        new->coords[i] = calloc(2, sizeof(float));
    }
    new->labels = calloc(new->firstdim, sizeof(int));
    int new_count = 0;


    for (int i = 0; i<temp->firstdim; i++){
        new->coords[new_count][0] = temp->coords[i][0];
        new->coords[new_count][1] = temp->coords[i][1];
        new->labels[new_count] = temp->labels[i];
        new_count++;
    }


    return new;

}
struct set * rename_set(struct set * temp){
    //tests all points with a given perceptron and dataset and returns the failures


    struct set * new = malloc(sizeof(struct set));
    new->firstdim = temp->firstdim;
    new->coords = malloc(new->firstdim * sizeof(float *));
    for (int i = 0; i<new->firstdim; i++){
        new->coords[i] = calloc(2, sizeof(float));
    }
    new->labels = calloc(new->firstdim, sizeof(int));
    int new_count = 0;


    for (int i = 0; i<temp->firstdim; i++){
        new->coords[new_count][0] = temp->coords[i][0];
        new->coords[new_count][1] = temp->coords[i][1];
        new->labels[new_count] = temp->labels[i];
        new_count++;
    }
    for (int i = 0; i<temp->firstdim; i++){
        free(temp->coords[i]);
    }
    free(temp->coords);
    free(temp->labels);
    free(temp);
    return new;

}
struct set * ret_tempf(struct perc * ada, struct set * temp){
    //tests all points with a given perceptron and dataset and returns the failures
    int fails = 0;
    for (int i = 0; i<temp->firstdim; i++){
        ada->inputs[0] = temp->coords[i][0];
        ada->inputs[1] = temp->coords[i][1];
        calc_perceptron_output(ada);
        perc_activation(ada);
        if (temp->labels[i] != ada->pred){
            fails++;
        }

    }
    if (fails == 0){
        return temp;
    }


    struct set * new = malloc(sizeof(struct set));
    new->firstdim = fails;
    new->coords = malloc(new->firstdim * sizeof(float *));
    for (int i = 0; i<new->firstdim; i++){
        new->coords[i] = calloc(2, sizeof(float));
    }
    new->labels = calloc(new->firstdim, sizeof(int));
    int new_count = 0;


    for (int i = 0; i<temp->firstdim; i++){
        ada->inputs[0] = temp->coords[i][0];
        ada->inputs[1] = temp->coords[i][1];
        calc_perceptron_output(ada);
        perc_activation(ada);
        if (ada->pred != temp->labels[i]){
            new->coords[new_count][0] = temp->coords[i][0];
            new->coords[new_count][1] = temp->coords[i][1];
            new->labels[new_count] = temp->labels[i];
            new_count++;
        }


    }
    for (int i = 0; i<temp->firstdim; i++){
        free(temp->coords[i]);
    }
    free(temp->coords);
    free(temp->labels);
    free(temp);
    return new;

}
struct set * rmv_point (struct set * temp, float x_max, float x_min){
    //removes a points from the data set and then returns new set should work?
    int survivors = 0;
    float highest = -FLT_MAX;
    //Calculate the midpoint between extremes on the lowest dimension (x)
    float mid_point = (x_max-x_min)/2;
    for (int i = 0; i < temp->firstdim; i++) {
        float x = temp->coords[i][0];
        if (!(x < x_max && x > mid_point)){
            continue;
        }
        else{
            survivors++;
        }

    }
    struct set * new = malloc(sizeof(struct set));
    new->firstdim = survivors;
    new->coords = malloc(new->firstdim * sizeof(float *));
    for (int i = 0; i<new->firstdim; i++){
        new->coords[i] = calloc(2, sizeof(float));
    }
    new->labels = calloc(new->firstdim, sizeof(int));
    int new_count = 0;


    for (int i = 0; i<temp->firstdim; i++){
        float x = temp->coords[i][0];
        if (!(x < x_max && x > mid_point)){

            continue;
        }
        else{
            new->coords[new_count][0] = temp->coords[i][0];
            new->coords[new_count][1] = temp->coords[i][1];
            new->labels[new_count] = temp->labels[i];
            new_count++;
        }
    }
    for (int i = 0; i<temp->firstdim; i++){
        free(temp->coords[i]);
    }
    free(temp->coords);
    free(temp->labels);
    free(temp);
    return new;

}
void free_set(struct set * temp){
    if (!temp) return;

    if (temp->coords){
        for (int i = 0; i < temp->firstdim; i++){
            free(temp->coords[i]);
        }
        free(temp->coords);
    }

    if (temp->labels){
        free(temp->labels);
    }

    free(temp);
}
void train_perc_stagsp (struct perc * ada, struct set * hold_subset){
    //This functions goal is to train the perceptron that is being bridged in to the network afterwards
    int stag_max = 200;
    int stag_count = 0;
    int false_count = 0;
    int false_hold = -INT_MAX;
    while (stag_count != stag_max){

        for (int i = 0; i<hold_subset->firstdim; i++){ //Training loop for depth 1
            ada->inputs[0] = hold_subset->coords[i][0];
            ada->inputs[1] = hold_subset->coords[i][1];
            calc_perceptron_output(ada);
            perc_activation(ada);
            if ((ada->pred > 0 && hold_subset->labels[i] < 0) || (ada->pred < 0 && hold_subset->labels[i] > 0)){ // Train when incorrect
                train_perc(ada, hold_subset->labels[i]);
                false_count++;

            }

        }
        if (false_count == false_hold){
            stag_count++;
        }
        if (false_count == 0){
            break;
        }
        false_hold = false_count;
        false_count = 0;
    }
}

struct perc * veri_perc_chain2(struct perc * ada, int firstdim, int seconddim, float data[firstdim][seconddim], int labels[firstdim], int depth){
    // This function is the only one needed to train a chain, this could definitely be done is a bst to be more efficient
    //Hold_subset will again be the defacto for handling training and such temp_f will carry over the points that need to be carried then assigned to hold_subset
    //Loops that will assign all values pertinent to the next training loop or testing loop to hold_subset
     struct perc * cur = ada;
    int any_fails = 0;

    struct set * hold_subset = calloc(1, sizeof(struct set));
    //Assign data values to hold_subset
    hold_subset->firstdim = firstdim;
    hold_subset->coords = malloc(hold_subset->firstdim * sizeof(float *));
    hold_subset->labels = calloc(hold_subset->firstdim, sizeof(int));
    for (int i = 0; i< hold_subset->firstdim; i++){
        hold_subset->coords[i] = calloc(2, sizeof(float));
        hold_subset->coords[i][0] = data[i][0];
        hold_subset->coords[i][1] = data[i][1];
        hold_subset->labels[i] = labels[i];

    }
    int fail_count = 0;
    int points_nothandled = 0;

    int in_bounds = 0;
    while(cur != NULL){
        
        struct set * pass_to_hold = calloc(1, sizeof(struct set));
        struct set * temp_f = calloc(1, sizeof(struct set));
        temp_f->firstdim = 0;
        fail_count = 0;
        points_nothandled = 0;

        in_bounds = 0;
        
        for (int i = 0; i<hold_subset->firstdim; i++){
            cur->inputs[0] = hold_subset->coords[i][0];
            cur->inputs[1] = hold_subset->coords[i][1];
            int bounds = (cur->max[0] > hold_subset->coords[i][0] && cur->min[0] < hold_subset->coords[i][0] && cur->max[1] > hold_subset->coords[i][1] && cur->min[1] < hold_subset->coords[i][1]);
            if ((bounds)){
                in_bounds++;
                calc_perceptron_output(cur);
                perc_activation(cur);
                if (hold_subset->labels[i] != cur->pred){
                    fail_count++;
                }
            }
            else{
                points_nothandled++;
            }
        }
        if (fail_count > 0){
            free_set(temp_f);
            temp_f = calloc(1, sizeof(struct set));
            

            temp_f->firstdim = fail_count;
            temp_f->labels = calloc(temp_f->firstdim, sizeof(int));
            temp_f->coords = malloc(temp_f->firstdim * sizeof(float *));
            for (int i = 0; i<temp_f->firstdim; i++){
                temp_f->coords[i] = calloc(2, sizeof(float));
            }
            //Put the failing points into temp_f
            int hold_fail = 0;

            for (int i = 0; i<hold_subset->firstdim; i++){
                cur->inputs[0] = hold_subset->coords[i][0];
                cur->inputs[1] = hold_subset->coords[i][1];
                int bounds = (cur->max[0] > hold_subset->coords[i][0] && cur->min[0] < hold_subset->coords[i][0] && cur->max[1] > hold_subset->coords[i][1] && cur->min[1] < hold_subset->coords[i][1]);
                if ((bounds)){

                    calc_perceptron_output(cur);
                    perc_activation(cur);
                    if (hold_subset->labels[i] != cur->pred){
                        temp_f->coords[hold_fail][0] = hold_subset->coords[i][0];
                        temp_f->coords[hold_fail][1] = hold_subset->coords[i][1];
                        temp_f->labels[hold_fail] = hold_subset->labels[i];
                        hold_fail++;
                    }
                }
            }
            //Now that there are values in the temp_f subset we can create and train the new perceptron and insert it
            printf("Temp_f firstdim: %d\n", temp_f->firstdim);
            struct perc * child = calloc(1, sizeof(struct perc));
            printf("Made child!");
            //MARK CLONE HANDLING
            perc_random_init(child);
            // smart_weight_init(child, temp_f);
            find_perc_bounds(child, temp_f);
            printf("Child Min X:%0.10f, Min Y:%0.10f, Max X:%0.10f, Max Y:%0.10f\n", child->min[0], child->min[1], child->max[0], child->max[1]);
            printf("Cur Min X:%0.10f, Min Y:%0.10f, Max X:%0.10f, Max Y:%0.10f\n", cur->min[0], cur->min[1], cur->max[0], cur->max[1]);
            int is_clone =
            float_equal(child->min[0], cur->min[0]) &&
            float_equal(child->min[1], cur->min[1]) &&
            float_equal(child->max[0], cur->max[0]) &&
            float_equal(child->max[1], cur->max[1]);

            if (is_clone){ //Needs rewrite to handle strange issues with explosion of perceptron chain
                //Child is clone
                printf("IS CLONE is not HANDLING CORRECT\n");
                printf("temp_f firstdim: %d\n", temp_f->firstdim);
                while (float_equal(child->max[0], cur->max[0])){
                    float hold_maxx = -FLT_MAX;
                    float hold_minx = FLT_MAX;
                    for (int i = 0; i<temp_f->firstdim; i++){
                        if (temp_f->coords[i][0] > hold_maxx){
                            hold_maxx = temp_f->coords[i][0];
                        }
                        if (temp_f->coords[i][0] < hold_minx){
                            hold_minx = temp_f->coords[i][0];
                        }
                    }
                    temp_f = rmv_point(temp_f, hold_maxx, hold_minx);
                    find_perc_bounds(child, temp_f);
                }
                printf("temp_f firstdim: %d\n", temp_f->firstdim);

            }
            find_perc_bounds(child, temp_f);
            train_perc_stagsp(child, temp_f);

            //Now for the insertion logic
            //Child needs to be cur->next, child->prev = cur, so on and so on
            if (cur->last == 1){
                cur->next = child;
                child->prev = cur;
                child->next = NULL;
                cur->last = 0;
                child->last = 1;
            }
            else{
                struct perc * next = cur->next;
                cur->next = child;
                child->next = next;
                child->prev = cur;
                if (next != NULL){
                    next->prev = child;
                }
                
            }
            free_set(temp_f);
            temp_f = NULL;

        }
        // No matter what we need to reduce size of set afterwards so all points in current set will go through this this just uses pass_to_hold to transfer relevant points not handled yet
        // On to hold_sub to be handled in the next iteration
        //Passing the points it didnt interact with that werent in bounds
        if (pass_to_hold != NULL){
            free_set(pass_to_hold);
            pass_to_hold = calloc(1, sizeof(struct set));
            pass_to_hold->coords = NULL;
            pass_to_hold->labels = NULL;
            pass_to_hold->firstdim = 0;

        }
        if (points_nothandled == 0){
            break; // nothing left to process
        }
        pass_to_hold->firstdim = points_nothandled;
        pass_to_hold->labels = calloc(pass_to_hold->firstdim, sizeof(int));
        pass_to_hold->coords = malloc(pass_to_hold->firstdim * sizeof(float *));
        for (int i = 0; i< pass_to_hold->firstdim; i++){
            pass_to_hold->coords[i] = calloc(2, sizeof(float));
        }

        //Point assignment
        int pass_idx = 0;
        for (int i = 0; i<hold_subset->firstdim; i++){
            cur->inputs[0] = hold_subset->coords[i][0];
            cur->inputs[1] = hold_subset->coords[i][1];
            int bounds = (cur->max[0] > hold_subset->coords[i][0] && cur->min[0] < hold_subset->coords[i][0] && cur->max[1] > hold_subset->coords[i][1] && cur->min[1] < hold_subset->coords[i][1]);
            if ((bounds)){
                continue;
            }
            else{
                pass_to_hold->coords[pass_idx][0] = hold_subset->coords[i][0];
                pass_to_hold->coords[pass_idx][1] = hold_subset->coords[i][1];
                pass_to_hold->labels[pass_idx] = hold_subset->labels[i];
                pass_idx++;
            }
        }
        free_set(hold_subset);
        hold_subset = pass_to_hold;//rename_set(pass_to_hold);
        pass_to_hold =NULL;


        cur = cur->prev;
        depth++;
    }

    while (cur->last != 1 && cur != NULL){
        cur = cur->next;
    }
    return cur;






    //So now we either have failure points or we dont if we do then the next step is to create a perceptron to train on those failure points
    // If we dont, then we just do nothing really besides traverse to the previous node and check there


}

struct perc * perc_full_init(){
    struct perc * ada = calloc(1, sizeof(struct perc));
    

}

//This section will be generally interacting with data whether in a file or tokenizing information
struct token * tokenizer(char * carry, int buf_size, char * alphabet, int alpha_len){
    struct token * token_list = malloc(sizeof(struct token) * buf_size);
    for (int i = 0; i<buf_size; i++){
        for (int j = 0; j<sizeof(alphabet); j++){
            if (carry[i] == alphabet[j]){
                token_list[i].idx = j;
                token_list[i].val = 1.0f;
            }
            if (i == buf_size-1){
                token_list[i].nextlidx = carry[0];
            }
            else{
                token_list[i].nextlidx = carry[i+1];
            }
        }
    }
    return token_list;

}
// This section is relevant for dealing with modern machine learning ideas such as the Artificial Neuron,
// the activation functions and other important modern topics including some theory
void hidden_math(struct node * alayer, struct node * blayer, int alen, int blen, int isout){

    for (int i = 0; i < blen; i++){
        blayer[i].value = 0.0f;
        for (int j = 0; j<alen; j++){
            if (alayer[j].active[i] != 0){ // Looks at weight mask to see if nodes are active
                blayer[i].value += alayer[j].weight[i] * alayer[j].value;
            }
        }
        if(!isout){
            blayer[i].value = tanh(blayer[i].value);
        }
        else{
            blayer[i].value = 1.0f / (1.0f + expf(-blayer[i].value)); //sigmoid out
        }

    }

}
void lth_hidden_math(struct node * alayer, struct node * blayer, int alen, int blen, int isout){

    for (int i = 0; i < blen; i++){
        blayer[i].value = 0.0f;
        for (int j = 0; j<alen; j++){
            if (alayer[j].weight[i] > 0.6f || alayer[j].weight[i] < -0.6f){
                blayer[i].value += alayer[j].weight[i] * alayer[j].value;
            }
        }
        if(!isout){
            blayer[i].value = tanh(blayer[i].value);
        }
        else{
            blayer[i].value = 1.0f / (1.0f + expf(-blayer[i].value)); //sigmoid out
        }

    }

}
float weighted_sum(float * a, float * b, int len){
    float out = 0.0;
    for(int i=0; i<len; i++){
        out += (a[i] * b[i]);
    }
    return out;

}
void final_layer_delta_calc(struct node * alayer, struct node * blayer, int alen, int blen){ // since the target and the output should always have the same amount of outputs I'm just going to use alen
    for (int i = 0; i<alen; i++){
        alayer[i].delta = blayer[i].value - alayer[i].value;
    }
}

void delta_calc(struct node * alayer, struct node * blayer, int alen, int blen) {
    for (int i = 0; i < alen; i++) {
        alayer[i].delta = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < blen; j++) {
            sum += blayer[j].delta * alayer[i].weight[j];


        }
        alayer[i].delta = sum * (1.0f - alayer[i].value * alayer[i].value);
    }

}

void back_prop(struct node * alayer, struct node * blayer, int alen, int blen, int cmp_f){
    for (int i = 0; i<alen; i++){
        for (int j = 0; j<blen; j++){
            if (alayer[i].active[j] != 0){ //if the mask at that node is zero ignore
                if (cmp_f == 1){ // This is going to be cool see Consensus Mask Pruning in readme
                    alayer[i].weight[j] -= .4 * blayer[j].delta * alayer[i].value;
                }
                else{
                    alayer[i].weight[j] -= 0.1 * blayer[j].delta * alayer[i].value;
                }
            }

        }

    }
    // weight = weight - learning_rate * (output - target) * input
}
void sigmoid(struct node * alayer, int alen, struct node *goals){
    for (int i = 0; i < alen; i++){
        float sig = alayer[i].value;
        alayer[i].delta = (alayer[i].value - goals[i].value) * sig * (1.0f - sig);
        // output->delta = (output->value - goals) * sig * (1.0f - sig);

    }

}
void clear_nodes(struct node * alayer, int alen){ // definitely might be faster to just hardcode but makes more sense for a demo
    for (int i = 0; i<alen; i++){
        alayer[i].value = 0.0f;
    }
}
void define_goals(struct node * alayer, struct node * blayer, int alen, int c_target){ // alayer is input blayer is goals

    if (c_target == 25){
        blayer[0].value = 1.0f;
    }
    else{
        blayer[c_target+1].value = 1.0f;
    }
}


float neural_net(float * inputs, float * weights){
    float pred = weighted_sum(inputs, weights, length);
    return pred;
}
float error(float pred, float true_w){
    return pow(pred - true_w, 2.0);
}

float delta(float pred, float true_g){
    return pred - true_g;
}

struct node * nodes_gen(int len, int next_layer_len){
    struct node * nodeinit = malloc(sizeof(struct node)*len);
    for(int i = 0; i < len; i++){
        for (int j = 0; j<next_layer_len; j++){
            float limit = sqrt(6.0f / (len + next_layer_len));
            nodeinit[i].weight[j] = ((float)rand() / RAND_MAX) * 2 * limit - limit;




            nodeinit[i].active[j] = 1;
        }
        nodeinit[i].value = 0.0f;
        nodeinit[i].delta = 0.0f;
        nodeinit[i].next = NULL;
    }
    return nodeinit;
}
void pruning(struct node * alayer, struct node * blayer, int alen, int blen, float minabsval){

    for (int i = 0; i < blen; i++){
        for (int j = 0; j<alen; j++){
            if (alayer[j].weight[i] > minabsval || alayer[j].weight[i] < -(minabsval)){
                continue;
            }
            else{
                alayer[j].active[i] = 0;
            }
        }

    }

}

int density(struct node * alayer, int alen, int blen){
    int n_zerod = 0;
    for(int i = 0; i<alen; i++){
        for (int j = 0; j<blen; j++){
            if (alayer[i].active[j] == 0){
                n_zerod++;
            }
        }
    }
    return n_zerod;
}

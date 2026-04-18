#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "structs.h"
#include "functions.h"


int main(){
    struct perc ada = {0};
    srand(time(NULL));
    perc_random_init(&ada);
    struct perc * first_case;
    struct perc placeholder = {0};
    struct perc *next;
    struct perc * list = &placeholder;
    
    //TWO MOONS DATASET
        float data[12][2] = {
        // Moon 1 (upper crescent)
        {-0.200,  0.000},
        {-0.150,  0.050},
        {-0.100,  0.080},
        {-0.050,  0.080},
        { 0.000,  0.040},
        { 0.050,  0.000},
        // Moon 2
        {-0.075,  0.025},
        {-0.050, -0.050},
        { 0.010, -0.080},
        { 0.060, -0.040},
        { 0.100,  0.000},
        { 0.125,  0.050}
    };
    int labels[12] = {1,1,1,1,1,1,-1,-1,-1,-1,-1,-1};
    
    float goals[2][2] = {{0,0},
                        {0,0}};
    float inputs[2][2] = {{0,0},
                          {0,0}};
    

    //////////////////
    //////////////////
    //TRAINING BLOCK//
    //////////////////
    //////////////////
    int stag_count = 0;
    int hold_false = INT_MIN;
    int false_count = 0;

    int i = 0;
    int depth = 0;
    struct set * hold_subset;
    hold_subset = malloc(sizeof(struct set));
    while (list->last != 1){
        printf("New perceptron\n");
        stag_count = 0; 
        hold_false = INT_MIN;
        struct perc * cur;
        if (depth == 0){
            cur = &ada;
            hold_subset->labels = labels;
            hold_subset->firstdim = 12;
            cur->prev = NULL;

        }
        else{
            cur = list;
        }
        struct set temp_s = {0}; // Temporary success set
        struct set temp_f = {0};
        int f_count = 0;
        int s_count = 0;
        int last_epoch = 0;
        while (stag_count <= 50){
            last_epoch = (stag_count == 50);
            for (int j = 0; j<hold_subset->firstdim; j++){
                for (int k = 0; k<2; k++){
                    if (depth == 0){
                        cur->inputs[k] = data[j][k];
                    }
                    else{
                        cur->inputs[k] = hold_subset->coords[j][k];
                    }
                    
                }
                calc_perceptron_output(cur);
                perc_activation(cur);
                
               
                if (cur->pred >= 0 && hold_subset->labels[j] >= 0 || cur->pred <= 0 && hold_subset->labels[j] <= 0){
                    printf("Successfully Predicted on %d turn!\n", i);

                }
                else{
                    int goal;
                    if (hold_subset->labels[j] > 0){
                        goal = 1;
                    }
                    else{
                        goal = -1;
                    }
                    printf("Learning is taking place!\n");
                    train_perc(cur, hold_subset->labels[j]);
                    false_count++;
                }
                
            
            }
            
            if (false_count == hold_false){
                stag_count++;

            }
            hold_false = false_count;
            false_count = 0;
            i++;

            if (last_epoch){
                    // This one is to get the dimensions for both the succesful set dimensions and the points that will continue on to be classified in further perceptrons
                    for (int j = 0; j<hold_subset->firstdim; j++){
                        for (int k = 0; k<2; k++){
                            if (depth == 0){
                                cur->inputs[k] = data[j][k];
                            }
                            else{
                                cur->inputs[k] = hold_subset->coords[j][k];
                            }
                            
                        }
                        calc_perceptron_output(cur);
                        perc_activation(cur);
                        if (cur->pred >= 0 && hold_subset->labels[j] >= 0 || cur->pred <= 0 && hold_subset->labels[j] <= 0){
                            printf("Successfully Predicted on %d turn!\n", i);
                            temp_s.firstdim++;
                            

                        }
                        else{
                            temp_f.firstdim++;
                            
                        }
                    }
                    
                    temp_s.seconddim = 2;
                    temp_f.seconddim = 2;
                    temp_s.coords = malloc(temp_s.firstdim * sizeof(float *));
                    temp_f.coords = malloc(temp_f.firstdim * sizeof(float *));
                    // if (depth > 0){
                    //     // for (int i = 0; i < 2; i++) {
                    //     // if (next->min[i] < cur->min[i]) next->min[i] = cur->min[i];
                    //     // if (next->max[i] > cur->max[i]) next->max[i] = cur->max[i];
                    // }
                    // }
                    
                    
                    // For saving the actual coordinates
                    s_count = 0;
                    f_count = 0;
                    for (int j = 0; j<hold_subset->firstdim; j++){
                        for (int k = 0; k<2; k++){
                            if (depth == 0){
                                cur->inputs[k] = data[j][k];
                            }
                            else{
                                cur->inputs[k] = hold_subset->coords[j][k];
                            }
                            
                        }
                        calc_perceptron_output(cur);
                        perc_activation(cur);
                        if (cur->pred >= 0 && hold_subset->labels[j] >= 0 || cur->pred <= 0 && hold_subset->labels[j] <= 0){
                            temp_s.coords[s_count] = malloc(2 * sizeof(float));
                            printf("Successfully Predicted on %d turn!\n", i);

                            if (depth == 0){
                                temp_s.coords[s_count] = data[j];
                            }
                            else{
                                temp_s.coords[s_count] = hold_subset->coords[j];
                            }
                            
                            s_count++;

                            
                            

                        }
                        else{
                            temp_f.coords[f_count] = malloc(2 * sizeof(float));
                            if (depth == 0){
                                temp_f.coords[f_count][0] = data[j][0];
                                temp_f.coords[f_count][1] = data[j][1];
                            }
                            else{
                                temp_f.coords[f_count][0] = hold_subset->coords[j][0];
                                temp_f.coords[f_count][1] = hold_subset->coords[j][1];
                            }
                            
                            

                            f_count++;
                            int goal;
                            if (hold_subset->labels[j] > 0){
                                goal = 1;
                            }
                            else{
                                goal = -1;
                            }
                            
                        }
                    }
                    
                    
                }
        }

        if (depth == 0){
            first_case = cur;


        }
        temp_f.labels = malloc(temp_f.firstdim * sizeof(int));
        for (int j = 0, fc = 0; j < hold_subset->firstdim; j++) {
            if (depth == 0){
                cur->inputs[0] = data[j][0];
                cur->inputs[1] = data[j][1];
            }
            else{
                cur->inputs[0] = hold_subset->coords[j][0];
                cur->inputs[1] = hold_subset->coords[j][1];
            }
            calc_perceptron_output(cur);
            perc_activation(cur);
            if (!((cur->pred >= 0 && hold_subset->labels[j] >= 0) || 
                (cur->pred <= 0 && hold_subset->labels[j] <= 0))) {
                temp_f.labels[fc++] = hold_subset->labels[j];
            }
        }
        hold_subset->coords = temp_f.coords;
        hold_subset->labels = temp_f.labels;
        hold_subset->firstdim = temp_f.firstdim;
        hold_subset->seconddim = 2;
        
       if (temp_f.firstdim > 0) {
            next = create_perc();
            next->prev = cur;
            cur->next = next;
            if (temp_s.coords != NULL){
                find_perc_bounds(cur, &temp_s);
            }
            
            find_perc_bounds(next, &temp_f);
            list = next;
            depth++;
            printf("Parent bounds: x[%.3f, %.3f] y[%.3f, %.3f]\n",
            cur->min[0], cur->max[0], cur->min[1], cur->max[1]);
            printf("Child bounds: x[%.3f, %.3f] y[%.3f, %.3f]\n",
                next->min[0], next->max[0], next->min[1], next->max[1]);
        } 
        else {
            // no failures, we're done
            find_perc_bounds(cur, &temp_s);
            cur->next = NULL;
            cur->last = 1;
            break;  // exit the outer while
        }
        printf("Depth %d complete. Failures: %d\n", depth, temp_f.firstdim);


        

    }
    
    int labels_set_one[6] = {1,1,1,1,1,1};
    int labels_set_two[6] = {-1,-1,-1,-1,-1,-1};
//Ultimately need to do one more test run through, would need do some verification to ensure that perceptron duplicates arent being made, and test for all the data that is within child bounds if no data has been 
// Will essentially do testing to determine if all points within max and min x and y will be successfully classified if yes great will proceed to testing otherwise will create perceptron to cover that area
// Reassurance Block
//Might not be necessary
// struct perc * asu;
// struct perc * hold;
// hold = &placeholder;
// asu = first_case;
// int iter = 0;
// struct set focus = {0};
// struct set suc = {0};
// struct set fail = {0};
// struct perc test; // Will be used to compare to all perceptrons to determine if the perceptron has already been made if so it will skip
// while(asu->next != NULL){
//     asu = hold;
//     if (iter == 0){
//         focus.firstdim = 12;
//         focus.seconddim = 2;
//         focus.coords = malloc(focus.firstdim * sizeof(float *));
//         for (int i = 0; i < focus.firstdim; i++) {
//             focus.coords[i] = malloc(focus.seconddim * sizeof(float));
//         };
//         for (int i = 0; i<focus.firstdim; i++){
//             focus.coords[i][0] = data[i][0];
//             focus.coords[i][1] = data[i][1];
//         }
//         for (int i = 0; i<focus.firstdim; i++){
//             asu->inputs[0] = focus.coords[i][0];
//             asu->inputs[1] = focus.coords[i][1];
//             calc_perceptron_output(asu);
//             perc_activation(asu);
//             if (asu->pred == labels[i]){
//                 suc.firstdim++;
//             }
//             else{
//                 fail.firstdim++;
//             }
//         }
//         suc.coords = malloc(suc.firstdim * sizeof(float*));
//         for (int i = 0; i < suc.firstdim; i++) {
//             suc.coords[i] = malloc(2 * sizeof(float));
//         }
//         suc.labels = malloc(suc.firstdim * sizeof(int));

//         fail.coords = malloc(fail.firstdim * sizeof(float*));
//         for (int i = 0; i < fail.firstdim; i++) {
//             fail.coords[i] = malloc(2 * sizeof(float));
//         }
//         fail.labels = malloc(fail.firstdim * sizeof(int));
//         int s_count = 0;
//         int f_count = 0;
//         for (int i = 0; i<focus.firstdim; i++){
//             asu->inputs[0] = focus.coords[i][0];
//             asu->inputs[1] = focus.coords[i][1];
//             calc_perceptron_output(asu);
//             perc_activation(asu);
//             if (asu->pred == labels[i]){
//                 suc.coords[s_count][0] = data[i][0];
//                 suc.coords[s_count][0] = data[i][1];
//                 suc.labels[s_count] = labels[i];
//                 s_count++;
//             }
//             else{
//                 fail.coords[f_count][0] = data[i][0];
//                 fail.coords[f_count][0] = data[i][1];
//                 fail.labels[f_count] = labels[i];
//                 f_count++;
//             }
//         }
//         find_perc_bounds(asu, &fail);
        
        

//     }
//     else{
//         focus.firstdim = fail.firstdim;
//         focus.seconddim = 2;
//         focus.coords = malloc(focus.firstdim * sizeof(float *));
//         for (int i = 0; i < focus.firstdim; i++) {
//             focus.coords[i] = malloc(focus.seconddim * sizeof(float));
//         }
//         focus.labels = malloc(focus.firstdim * sizeof(int));
//         for (int i = 0; i<focus.firstdim; i++){
//             focus.coords[i][0] = fail.coords[i][0];
//             focus.coords[i][1] = fail.coords[i][1];
//             focus.labels[i] = fail.labels[i];
//         }
//         for (int i = 0; i<focus.firstdim; i++){
//             asu->inputs[0] = focus.coords[i][0];
//             asu->inputs[1] = focus.coords[i][1];
//             calc_perceptron_output(asu);
//             perc_activation(asu);
//             if (asu->pred == fail.labels[i]){
//                 suc.firstdim++;
//             }
//             else{
//                 fail.firstdim++;
//             }
//         }
//         suc.coords = malloc(suc.firstdim * sizeof(float*));
//         for (int i = 0; i < suc.firstdim; i++) {
//             suc.coords[i] = malloc(2 * sizeof(float));
//         }

//         fail.coords = malloc(fail.firstdim * sizeof(float*));
//         for (int i = 0; i < fail.firstdim; i++) {
//             fail.coords[i] = malloc(2 * sizeof(float));
//         }
//         suc.labels = malloc(suc.firstdim * sizeof(int));
//         fail.labels = malloc(fail.firstdim * sizeof(int));
//         int s_count = 0;
//         int f_count = 0;
//         for (int i = 0; i<focus.firstdim; i++){
//             asu->inputs[0] = focus.coords[i][0];
//             asu->inputs[1] = focus.coords[i][1];
//             calc_perceptron_output(asu);
//             perc_activation(asu);
//             if (asu->pred == fail.labels[i]){
//                 suc.coords[s_count][0] = focus.coords[i][0];
//                 suc.coords[s_count][1] = focus.coords[i][1];
//                 s_count++;
//             }
//             else{
//                 fail.coords[f_count][0] = focus.coords[i][0];
//                 fail.coords[f_count][1] = focus.coords[i][1];
//                 f_count++;
//             }
//         }
//         find_perc_bounds(asu, &fail);
        

//     }
//     hold = list;
//     if ((asu->min[0] == hold->min[0] && asu->min[1] == hold->min[1]) && asu->max[0] == hold->max[0] && asu->max[1] == hold->max[1]){
//         continue;
//     }
//     else{
//         if (hold->next != NULL){
//            
//         }
//     }
    

//     iter++;
//     break;

// }
printf("Finished Verification Successfully");


    //////////////////
    //////////////////
    // TESTING BLOCK//
    //////////////////
    //////////////////
   printf("=== TESTING BLOCK (CASCADE) v2 ===\n");
int correct = 0;
int total = 12;
// After training, before testing block

for (int j = 0; j < total; j++) {
    struct perc *trav = list;  // start at deepest perceptron
    int depth_used = depth;
    int pred = perceptron_trav(trav, data[j][0],data[j][1], 0.000001, 0);
    printf("Point %d: pred=%d label=%d %s\n", j, pred, labels[j], pred==labels[j] ? "OK" : "WRONG");
    if (pred == labels[j]){
        correct++;
    }

}

//    

printf("\nFinal Accuracy: %d/%d (%.2f%%)\n",
       correct, total, 100.0f * correct / total);
    
    return 0;
}
//MAYBE
//Instead of relying on it to classify a certain number of points on both correctly optimize it to get the most of a single group divided from the rest,
//This method would be a bit more complicated to implement but would work to classify then iterate or find the next nearest failing point in every direction 
// Certainly another approach if this fails 


//  while (trav != NULL) {
//         trav->inputs[0] = data[j][0];
//         trav->inputs[1] = data[j][1];

//         calc_perceptron_output(trav);
//         perc_activation(trav);

//         int correct_pred = (trav->pred == 1 && labels[j] == 1) ||
//                            (trav->pred == -1 && labels[j] == -1);

//         int in_bounds =
//             (data[j][0] >= trav->min[0] && data[j][0] <= trav->max[0]) &&
//             (data[j][1] >= trav->min[1] && data[j][1] <= trav->max[1]);

//         // If prediction is wrong or point is out of bounds, go to previous perceptron
//         if (!correct_pred || !in_bounds) {
//             trav = trav->prev;
//             depth_used--;
//             continue;
//         }

//         // Stop at first perceptron that correctly classifies
//         break;
//     }

//     if (trav == NULL) {
//         // fallback first perceptron
//         trav = list;
//         while (trav->prev != NULL) trav = trav->prev;
//         trav->inputs[0] = data[j][0];
//         trav->inputs[1] = data[j][1];
//         calc_perceptron_output(trav);
//         perc_activation(trav);
//     }

//     int final_correct = (trav->pred == 1 && labels[j] == 1) ||
//                         (trav->pred == -1 && labels[j] == -1);
//     if (final_correct) correct++;

//     printf("Point %2d [%6.3f, %6.3f] -> pred: %2d  label: %2d  %s (depth %d)\n",
//            j, data[j][0], data[j][1],
//            trav->pred, labels[j],
//            final_correct ? "CORRECT" : "WRONG",
//            depth_used);
// }
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "structs.h"
#include "functions.h"


int main(){
    struct perc ada = {0};
    srand(time(NULL));
    perc_random_init(&ada);
    struct perc * first_case;
    struct perc placeholder = {0};
    struct perc *next;
    struct perc * list = &placeholder;
    
    //TWO MOONS DATASET
        float data[12][2] = {
        // Moon 1 (upper crescent)
        {-0.200,  0.000},
        {-0.150,  0.050},
        {-0.100,  0.080},
        {-0.050,  0.080},
        { 0.000,  0.040},
        { 0.050,  0.000},
        // Moon 2
        {-0.075,  0.025},
        {-0.050, -0.050},
        { 0.010, -0.080},
        { 0.060, -0.040},
        { 0.100,  0.000},
        { 0.125,  0.050}
    };

    
    float goals[2][2] = {{0,0},
                        {0,0}};
    float inputs[2][2] = {{0,0},
                          {0,0}};
    int labels[12] = {1,1,1,1,1,1,-1,-1,-1,-1,-1,-1};

    //////////////////
    //////////////////
    //TRAINING BLOCK//
    //////////////////
    //////////////////
    int stag_count = 0;
    int hold_false = INT_MIN;
    int false_count = 0;

    int i = 0;
    int depth = 0;
    struct set * hold_subset;
    hold_subset = malloc(sizeof(struct set));
    while (list->last != 1){
        printf("New perceptron\n");
        stag_count = 0; 
        hold_false = INT_MIN;
        struct perc * cur;
        if (depth == 0){
            cur = &ada;
            hold_subset->labels = labels;
            hold_subset->firstdim = 12;
            cur->prev = NULL;

        }
        else{
            cur = list;
        }
        struct set temp_s = {0}; // Temporary success set
        struct set temp_f = {0};
        int f_count = 0;
        int s_count = 0;
        int last_epoch = 0;
        while (stag_count <= 100){
            last_epoch = (stag_count == 100);
            for (int j = 0; j<hold_subset->firstdim; j++){
                for (int k = 0; k<2; k++){
                    if (depth == 0){
                        cur->inputs[k] = data[j][k];
                    }
                    else{
                        cur->inputs[k] = hold_subset->coords[j][k];
                    }
                    
                }
                calc_perceptron_output(cur);
                perc_activation(cur);
                
               
                if (cur->pred >= 0 && hold_subset->labels[j] >= 0 || cur->pred <= 0 && hold_subset->labels[j] <= 0){
                    printf("Successfully Predicted on %d turn!\n", i);

                }
                else{
                    int goal;
                    if (hold_subset->labels[j] > 0){
                        goal = 1;
                    }
                    else{
                        goal = -1;
                    }
                    printf("Learning is taking place!\n");
                    train_perc(cur, hold_subset->labels[j]);
                    false_count++;
                }
                
            
            }
            
            if (false_count == hold_false){
                stag_count++;

            }
            hold_false = false_count;
            false_count = 0;
            i++;

            if (last_epoch){
                    // This one is to get the dimensions for both the succesful set dimensions and the points that will continue on to be classified in further perceptrons
                    temp_s.firstdim = 0;
                    temp_f.firstdim = 0;
                    s_count = 0;
                    f_count = 0;
                    for (int j = 0; j<hold_subset->firstdim; j++){
                        for (int k = 0; k<2; k++){
                            if (depth == 0){
                                cur->inputs[k] = data[j][k];
                            }
                            else{
                                cur->inputs[k] = hold_subset->coords[j][k];
                            }
                            
                        }
                        calc_perceptron_output(cur);
                        perc_activation(cur);
                        if (cur->pred >= 0 && hold_subset->labels[j] >= 0 || cur->pred <= 0 && hold_subset->labels[j] <= 0){
                            printf("Successfully Predicted on %d turn!\n", i);
                            temp_s.firstdim++;
                            

                        }
                        else{
                            temp_f.firstdim++;
                            
                        }
                    }
                    
                    temp_s.seconddim = 2;
                    temp_f.seconddim = 2;
                    temp_s.coords = malloc(temp_s.firstdim * sizeof(float *));
                    temp_f.coords = malloc(temp_f.firstdim * sizeof(float *));
                    // if (depth > 0){
                    //     // for (int i = 0; i < 2; i++) {
                    //     // if (next->min[i] < cur->min[i]) next->min[i] = cur->min[i];
                    //     // if (next->max[i] > cur->max[i]) next->max[i] = cur->max[i];
                    // }
                    // }
                    
                    
                    // For saving the actual coordinates
                    s_count = 0;
                    f_count = 0;
                    for (int j = 0; j<hold_subset->firstdim; j++){
                        for (int k = 0; k<2; k++){
                            if (depth == 0){
                                cur->inputs[k] = data[j][k];
                            }
                            else{
                                cur->inputs[k] = hold_subset->coords[j][k];
                            }
                            
                        }
                        calc_perceptron_output(cur);
                        perc_activation(cur);
                        if (cur->pred >= 0 && hold_subset->labels[j] >= 0 || cur->pred <= 0 && hold_subset->labels[j] <= 0){
                            temp_s.coords[s_count] = malloc(2 * sizeof(float));
                            if (depth == 0){
                                temp_s.coords[s_count][0] = data[j][0];
                                temp_s.coords[s_count][1] = data[j][1];
                            }
                            else{
                                temp_s.coords[s_count][0] = hold_subset->coords[j][0];
                                temp_s.coords[s_count][1] = hold_subset->coords[j][1];
                            }
                            s_count++;
                        }
                        else{
                            temp_f.coords[f_count] = malloc(2 * sizeof(float));
                            if (depth == 0){
                                temp_f.coords[f_count][0] = data[j][0];
                                temp_f.coords[f_count][1] = data[j][1];
                            }
                            else{
                                temp_f.coords[f_count][0] = hold_subset->coords[j][0];
                                temp_f.coords[f_count][1] = hold_subset->coords[j][1];
                            }
                            
                            

                            f_count++;
                            int goal;
                            if (hold_subset->labels[j] > 0){
                                goal = 1;
                            }
                            else{
                                goal = -1;
                            }
                            
                        }
                    }
                    
                    
                }
        }

        if (depth == 0){
            first_case = cur;


        }
        temp_f.labels = malloc(temp_f.firstdim * sizeof(int));
        for (int j = 0, fc = 0; j < hold_subset->firstdim; j++) {
            if (depth == 0){
                cur->inputs[0] = data[j][0];
                cur->inputs[1] = data[j][1];
            }
            else{
                cur->inputs[0] = hold_subset->coords[j][0];
                cur->inputs[1] = hold_subset->coords[j][1];
            }
            calc_perceptron_output(cur);
            perc_activation(cur);
            if (!((cur->pred >= 0 && hold_subset->labels[j] >= 0) || 
                (cur->pred <= 0 && hold_subset->labels[j] <= 0))) {
                temp_f.labels[fc++] = hold_subset->labels[j];
            }
        }
        hold_subset->coords = temp_f.coords;
        hold_subset->labels = temp_f.labels;
        hold_subset->firstdim = temp_f.firstdim;
        hold_subset->seconddim = 2;
        
       if (temp_f.firstdim > 0) {
            next = create_perc();
            next->prev = cur;
            cur->next = next;
            if (temp_s.coords != NULL){
                find_perc_bounds(cur, &temp_s);
            }
            
            find_perc_bounds(next, &temp_f);
            list = next;
            depth++;
            printf("Parent bounds: x[%.3f, %.3f] y[%.3f, %.3f]\n",
            cur->min[0], cur->max[0], cur->min[1], cur->max[1]);
            printf("Child bounds: x[%.3f, %.3f] y[%.3f, %.3f]\n",
                next->min[0], next->max[0], next->min[1], next->max[1]);
        } 
        else {
            // no failures, we're done
            find_perc_bounds(cur, &temp_s);
            cur->next = NULL;
            cur->last = 1;
            break;  // exit the outer while
        }
        printf("Depth %d complete. Failures: %d\n", depth, temp_f.firstdim);


        

    }
    
    int labels_set_one[6] = {1,1,1,1,1,1};
    int labels_set_two[6] = {-1,-1,-1,-1,-1,-1};
//Ultimately need to do one more test run through, would need do some verification to ensure that perceptron duplicates arent being made, and test for all the data that is within child bounds if no data has been 
// Will essentially do testing to determine if all points within max and min x and y will be successfully classified if yes great will proceed to testing otherwise will create perceptron to cover that area
// Reassurance Block
//Might not be necessary


// }
printf("Finished Verification Successfully");


    //////////////////
    //////////////////
    // TESTING BLOCK//
    //////////////////
    //////////////////
   printf("=== TESTING BLOCK (CASCADE) v2 ===\n");
int correct = 0;
int total = 12;
// After training, before testing block

for (int j = 0; j < total; j++) {
    struct perc *trav = list;  // start at deepest perceptron
    int depth_used = depth;
    int pred = perceptron_trav(trav, data[j][0],data[j][1], 0.000001, 0);
    printf("Point %d: pred=%d label=%d %s\n", j, pred, labels[j], pred==labels[j] ? "OK" : "WRONG");
    if (pred == labels[j]){
        correct++;
    }

}

//    

printf("\nFinal Accuracy: %d/%d (%.2f%%)\n",
       correct, total, 100.0f * correct / total);
    
    return 0;
}
//MAYBE
//Instead of relying on it to classify a certain number of points on both correctly optimize it to get the most of a single group divided from the rest,
//This method would be a bit more complicated to implement but would work to classify then iterate or find the next nearest failing point in every direction 
// Certainly another approach if this fails 
//  while (trav != NULL) {
//         trav->inputs[0] = data[j][0];
//         trav->inputs[1] = data[j][1];

//         calc_perceptron_output(trav);
//         perc_activation(trav);

//         int correct_pred = (trav->pred == 1 && labels[j] == 1) ||
//                            (trav->pred == -1 && labels[j] == -1);

//         int in_bounds =
//             (data[j][0] >= trav->min[0] && data[j][0] <= trav->max[0]) &&
//             (data[j][1] >= trav->min[1] && data[j][1] <= trav->max[1]);

//         // If prediction is wrong or point is out of bounds, go to previous perceptron
//         if (!correct_pred || !in_bounds) {
//             trav = trav->prev;
//             depth_used--;
//             continue;
//         }

//         // Stop at first perceptron that correctly classifies
//         break;
//     }

//     if (trav == NULL) {
//         // fallback first perceptron
//         trav = list;
//         while (trav->prev != NULL) trav = trav->prev;
//         trav->inputs[0] = data[j][0];
//         trav->inputs[1] = data[j][1];
//         calc_perceptron_output(trav);
//         perc_activation(trav);
//     }

//     int final_correct = (trav->pred == 1 && labels[j] == 1) ||
//                         (trav->pred == -1 && labels[j] == -1);
//     if (final_correct) correct++;

//     printf("Point %2d [%6.3f, %6.3f] -> pred: %2d  label: %2d  %s (depth %d)\n",
//            j, data[j][0], data[j][1],
//            trav->pred, labels[j],
//            final_correct ? "CORRECT" : "WRONG",
//            depth_used);
// }
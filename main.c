#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "structs.h"
#include "functions.h"


int main(){
    srand(time(NULL));

    //Just an XOR Data Set
    float data[40][2] = {
        { 0.05f, 0.80f }, { 0.10f, 0.90f }, { 0.15f, 0.85f }, { 0.20f, 0.75f },
        { 0.08f, 0.95f }, { 0.12f, 0.78f }, { 0.18f, 0.92f }, { 0.22f, 0.88f },
        { 0.06f, 0.82f }, { 0.25f, 0.80f },
        { 0.75f, 0.05f }, { 0.80f, 0.10f }, { 0.85f, 0.15f }, { 0.90f, 0.08f },
        { 0.78f, 0.20f }, { 0.82f, 0.12f }, { 0.88f, 0.18f }, { 0.92f, 0.22f },
        { 0.76f, 0.06f }, { 0.95f, 0.10f },
        { 0.75f, 0.80f }, { 0.80f, 0.90f }, { 0.85f, 0.85f }, { 0.90f, 0.75f },
        { 0.78f, 0.95f }, { 0.82f, 0.78f }, { 0.88f, 0.92f }, { 0.92f, 0.88f },
        { 0.76f, 0.82f }, { 0.95f, 0.80f },
        { 0.05f, 0.05f }, { 0.10f, 0.10f }, { 0.15f, 0.08f }, { 0.08f, 0.18f },
        { 0.12f, 0.22f }, { 0.20f, 0.12f }, { 0.18f, 0.06f }, { 0.22f, 0.15f },
        { 0.06f, 0.12f }, { 0.25f, 0.08f },
    };
    int labels[40] = {
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };

    // Initialization of perceptron and base set etc
    struct perc * list = create_perc();
    list->last = 1;
    struct set * base = create_subset(40, 2, data, labels);
    smart_weight_init(list, base);
    find_perc_bounds(list, base);
    list = veri_perc_chain2(list, 40, 2, data, labels, 0);
    
    if (list->last != 1){
        printf("Didnt return final node\n");
        return -1;
    }
    struct perc * test = list;
    test = list;
    while (test != NULL){
        printf("Is Clone: %d\n", test->clone);
        printf("Min X:%0.8f, Min Y:%0.8f, Max X:%0.8f, Max Y:%0.8f\n", test->min[0], test->min[1], test->max[0], test->max[1]);
        test = test->prev;
    }

    ////////////////////
    ////////////////////
    //// TESTING BLOCK//
    ////////////////////
    ////////////////////
    int depth = 0;
    int correct = 0;
    int total = 40;
    int depth_2 = 0;
    while (correct != total){
        int cata_err = 0;
        for (int j = 0; j < total; j++) {
            struct perc *trav = list;  // start at deepest perceptron
            int depth_used = depth;
            int pred = perceptron_trav(trav, data[j][0],data[j][1], 0.000001, 0);
            // printf("Point %d: pred=%d label=%d %s\n", j, pred, labels[j], pred==labels[j] ? "OK" : "WRONG");
            if (pred == labels[j]){
                correct++;
            }
            

        }
        if (correct == total){
            break;
        }
        
        printf("%d correct out of %d\n", correct, total);
        list = veri_perc_chain2(list, 40, 2, data, labels, depth_2);
        printf("Returned!\n");
        if (list->weights[0] == 404.0f){
            while (list != NULL){
                printf("Min X:%0.10f, Min Y:%0.10f, Max X:%0.10f, Max Y:%0.10f\n", list->min[0], list->min[1], list->max[0], list->max[1]);
                list = list->prev;
                
            }
            
            return -1;
        }
        correct = 0;
    }

    correct = 0;
   printf("=== TESTING BLOCK (CASCADE) v2 ===\n");

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
    printf("\nFinal Accuracy: %d/%d (%.2f%%)\n",
        correct, total, 100.0f * correct / total);
    test = list;

    free_set(base);
}




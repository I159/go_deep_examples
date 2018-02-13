package main

import (
	"fmt"
	"log"

	"github.com/I159/go_deep"
)

func main() {
	tLabels, err := getMNISTTrainingLabels("t10k-labels-idx1-ubyte", 10)
	if err != nil {
		log.Fatal(err)
	}
	labels, err := getMNISTTrainingLabels("train-labels-idx1-ubyte", 10)
	if err != nil {
		log.Fatal(err)
	}

	tSet, err := getMNISTTrainingImgs("t10k-images-idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}
	set, err := getMNISTTrainingImgs("train-images-idx3-ubyte")
	if err != nil {
		log.Fatal(err)
	}

	nn := go_deep.NewPerceptron(.01, &go_deep.Sygmoid{}, &go_deep.Quadratic{}, 784, 64, 10, 512)

	learnCost := nn.Learn(set, labels)
	for _, i := range learnCost {
		fmt.Println(i)
	}

	accuracy, _ := nn.Measure(tSet, tLabels)
	fmt.Printf("Accuracy: %f\n", accuracy)
}

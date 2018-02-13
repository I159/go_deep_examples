package main

import (
	"fmt"
	"log"

	"github.com/I159/go_deep"
	tm "github.com/buger/goterm"
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

	nn := go_deep.NewPerceptron(.00001, &go_deep.Sygmoid{}, &go_deep.Quadratic{}, 784, 64, 10, 8, 64)

	learnCost := nn.Learn(set, labels)

	chart := tm.NewLineChart(100, 20)
	data := new(tm.DataTable)
	data.AddColumn("Time")
	data.AddColumn("Cost")
	for i, c := range learnCost {
		data.AddRow(float64(i/10), c)
	}

	accuracy, _ := nn.Measure(tSet, tLabels)
	fmt.Printf("Accuracy: %f\n", accuracy)
	tm.Println(chart.Draw(data))
	tm.Flush()
}

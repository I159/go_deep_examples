package main

import (
	"fmt"
	"log"
	"math"

	"github.com/I159/go_deep"
	tm "github.com/buger/goterm"
)

const TOLERANCE = 0.000001

func equal(a, b float64) bool {
	if diff := a - b; diff < TOLERANCE {
		return true
	}
	return false
}

func main() {
	//tLabels, err := getMNISTTrainingLabels("t10k-labels-idx1-ubyte", 10)
	//if err != nil {
		//log.Fatal(err)
	//}
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

	nn := go_deep.NewPerceptron(.0001, &go_deep.Sygmoid{}, &go_deep.Quadratic{}, 784, 64, 10, 16, 764)

	learnCost := nn.Learn(set, labels)

	chart := tm.NewLineChart(100, 20)
	data := new(tm.DataTable)
	data.AddColumn("Time")
	data.AddColumn("Cost")
	for i, c := range learnCost {
		data.AddRow(float64(i/10), c)
	}
	tm.Println(chart.Draw(data))

	prediction := nn.Recognize(tSet)
	for _, pred := range prediction {
		max := 0.0
		for _, p := range pred {
			fmt.Println(p)
			local := math.Max(max, p)
			if !equal(local, max) {
				max = local
			}
		}
		fmt.Printf("MAX: %f", max)
		fmt.Println("~~~~~~~~~")
	}

	tm.Flush()
}

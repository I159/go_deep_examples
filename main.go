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

	shape := go_deep.Shape{
		InputSize: 784,
		HiddenSizes: []int{64},
		OutputSize: 10,
		HiddenLearningRates: []float64{0.001},
		HiddenActivations: []activation{&go_deep.Sygmoid{}},
		OutputActivation: &go_deep.Sygmoid{},
		Cost: &go_deep.Quadratic{},
	}
	nn := go_deep.NewPerceptron(shape)

	learnCost := nn.Learn(set, labels)

	chart := tm.NewLineChart(100, 20)
	data := new(tm.DataTable)
	data.AddColumn("Time")
	data.AddColumn("Cost")
	for i, c := range learnCost {
		data.AddRow(float64(i/10), c)
	}
	tm.Println(chart.Draw(data))

	accuracy := map[bool]float64{true: 0, false: 0}
	prediction := nn.Recognize(tSet)
	for i, pred := range prediction {
		max := 0.0
		maxIdx := 0
		for j, p := range pred {
			fmt.Println(p)
			local := math.Max(max, p)
			if !equal(local, max) {
				max = local
				maxIdx = j
			}
		}
		accuracy[tLabels[i][maxIdx] == 1]++
		fmt.Printf("MAX: %f IDX: %d, LABEL: %.0f\n", max, maxIdx, tLabels[i][maxIdx])
		fmt.Println("~~~~~~~~~")
	}

	tm.Flush()
	fmt.Printf("Accuracy: %f\n", accuracy[true] / accuracy[false])
}

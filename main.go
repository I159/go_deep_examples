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

	inputShape := go_deep.InputShape{
		Size: 784,
		LearningRate: .001,
	}
	hiddenLayers := []go_deep.HiddenShape{
		go_deep.HiddenShape{
			Size: 64,
			LearningRate: .001,
			Activation: new(go_deep.Sygmoid),
		},
	}
	outputLayer := go_deep.OutputShape{
		Size: 10,
		Activation: new(go_deep.Sygmoid),
		Cost: new(go_deep.Quadratic),
	}
	nn := go_deep.NewPerceptron(inputShape, hiddenLayers, outputLayer)

	learnCost := nn.Learn(set, labels, 4, 1024)

	chart := tm.NewLineChart(100, 20)
	data := new(tm.DataTable)
	data.AddColumn("Time")
	data.AddColumn("Cost")
	for i, c := range learnCost {
		data.AddRow(float64(i), c)
	}
	tm.Println(chart.Draw(data))

	//accuracy := map[bool]float64{true: 0, false: 0}
	prediction := nn.Recognize(tSet)
	for i, pred := range prediction {
		max := 0.0
		idx := 0
		label := 0
		for j, p := range pred {
			//fmt.Println(p, tLabels[i][j])

			local := math.Max(max, p)
			if !equal(local, max) {
				max = local
				idx = j
			}
			if tLabels[i][j] == 1 {
				label = j
			}
		}
		fmt.Printf("PREDICTION: %d LABEL: %d\n", idx, label)
	}

	tm.Flush()
	//fmt.Printf("Accuracy: %f\n", accuracy[true] / accuracy[false])
}

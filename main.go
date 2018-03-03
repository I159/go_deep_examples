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

func getSets() (set, tSet, labels, tLabels [][]float64, err error) {
	tLabels, err = getMNISTTrainingLabels("t10k-labels-idx1-ubyte", 10)
	if err != nil {
		return
	}
	labels, err = getMNISTTrainingLabels("train-labels-idx1-ubyte", 10)
	if err != nil {
		return
	}

	tSet, err = getMNISTTrainingImgs("t10k-images-idx3-ubyte")
	if err != nil {
		return
	}
	for i := range tSet {
		for j := range tSet[i] {
			if tSet[i][j] == 0 {
				tSet[i][j] = 1.
			}
		}
	}
	set, err = getMNISTTrainingImgs("train-images-idx3-ubyte")
	if err != nil {
		return
	}
	for i := range set {
		for j := range set[i] {
			if set[i][j] == 0 {
				set[i][j] = 1.
			}
		}
	}
	return
}

func declareNetwork() go_deep.Network {
	inputShape := go_deep.InputShape{
		Size:         784,
		LearningRate: .001,
	}
	hiddenLayers := []go_deep.HiddenShape{
		go_deep.HiddenShape{
			Size:         64,
			LearningRate: .001,
			Bias: 0.5,
			Activation:  go_deep.NewSigmoid(784, [2]float64{0, 256}, [2]float64{-0.5, 0.5}, 0),
		},
	}
	outputLayer := go_deep.OutputShape{
		Size:       10,
		Activation: go_deep.NewSigmoid(64, [2]float64{0, 1}, [2]float64{-1, 1}, 0.5),
		Cost:       new(go_deep.Quadratic),
	}
	return go_deep.NewPerceptron(inputShape, hiddenLayers, outputLayer)
}

func visualizeGradient(learnCost []float64) {
	chart := tm.NewLineChart(99, 20)
	data := new(tm.DataTable)
	data.AddColumn("Time")
	data.AddColumn("Cost")
	for i, c := range learnCost {
		data.AddRow(float64(i), c)
	}
	tm.Println(chart.Draw(data))
	tm.Flush()
}

func countAccuracy(prediction, tLabels [][]float64) {
	//accuracy := map[bool]float64{true: 0, false: 0}
	for i, pred := range prediction {
		max := 0.0
		idx := 0
		label := 0
		for j, p := range pred {
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
	//fmt.Printf("Accuracy: %f\n", accuracy[true] / accuracy[false])
}

func main() {
	set, tSet, labels, tLabels, err := getSets()
	if err != nil {
		log.Fatal(err.Error())
	}

	nn := declareNetwork()

	learnCost, err := nn.Learn(set, labels, 1, 1024)
	if err != nil {
		log.Fatal(err.Error())
	}

	prediction, err := nn.Recognize(tSet)
	if err != nil {
		log.Fatal(err.Error())
	}

	countAccuracy(prediction, tLabels)
	visualizeGradient(learnCost)
}

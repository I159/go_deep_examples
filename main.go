package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/I159/go_deep"
	tm "github.com/buger/goterm"
	"github.com/kevin-cantwell/dotmatrix"
)

const TOLERANCE = 0.000001

func equal(a, b float64) bool {
	if diff := a - b; diff < TOLERANCE {
		return true
	}
	return false
}

func getSets(path string) (set, tSet, labels, tLabels [][]float64, err error) {
	tLabels, err = getMNISTTrainingLabels(filepath.Join(path, "t10k-labels-idx1-ubyte"), 10)
	if err != nil {
		return
	}
	labels, err = getMNISTTrainingLabels(filepath.Join(path, "train-labels-idx1-ubyte"), 10)
	if err != nil {
		return
	}

	tSet, err = getMNISTTrainingImgs(filepath.Join(path, "t10k-images-idx3-ubyte"))
	if err != nil {
		return
	}
	for i := range tSet {
		for j := range tSet[i] {
			tSet[i][j] = (tSet[i][j] - 127.5) / 127.5
		}
	}

	set, err = getMNISTTrainingImgs(filepath.Join(path, "train-images-idx3-ubyte"))
	if err != nil {
		return
	}
	for i := range set {
		for j := range set[i] {
			set[i][j] = (set[i][j] - 127.5) / 127.5
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
			Bias:         0.5,
			Activation:   new(go_deep.Sigmoid),
		},
	}
	outputLayer := go_deep.OutputShape{
		Size:       10,
		Activation: new(go_deep.Sigmoid),
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

func countAccuracy(prediction, tLabels, set [][]float64) {
	accuracyMax := map[bool]float64{true: 0, false: 0}
	for i, pred := range prediction {
		max := 0.0
		maxIdx := 0
		label := 0
		for j, p := range pred {
			localMax := math.Max(max, p)
			if !equal(localMax, max) {
				max = localMax
				maxIdx = j
			}
			if tLabels[i][j] == 1 {
				label = j
			}
		}
		img := image.NewGray(image.Rect(0, 0, 28, 28))

		// Revert image pixels to uint8 value
		var pix []uint8
		for _, k := range set[i] {
			pix = append(pix, uint8(k*127.5+127.5))
		}

		img.Pix = pix
		dotmatrix.Print(os.Stdout, img)
		fmt.Printf("MAX: %d LABEL: %d\n", maxIdx, label)
		accuracyMax[maxIdx == label]++
	}
	fmt.Printf("Accuracy: %f\n", accuracyMax[true]/accuracyMax[false])
}

func main() {
	epochs := flag.Int("epochs", 1, "Number of epochs of learning")
	batch := flag.Int("batch", 512, "Batch size in items")
	path := flag.String("path", "~/Downloads", "Path to a directory containing binary data set files")
	flag.Parse()

	set, tSet, labels, tLabels, err := getSets(*path)
	if err != nil {
		log.Fatal(err.Error())
	}

	nn := declareNetwork()

	learnCost, err := nn.Learn(set, labels, *epochs, *batch)
	if err != nil {
		log.Fatal(err.Error())
	}

	prediction, err := nn.Recognize(tSet)
	if err != nil {
		log.Fatal(err.Error())
	}

	countAccuracy(prediction, tLabels, tSet)
	visualizeGradient(learnCost)
}

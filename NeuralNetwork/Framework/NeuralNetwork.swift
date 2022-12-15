//
//  NeuralNetwork.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation
import SwiftMatrix
import Accelerate

// Training video:
// https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1235s
struct NeuralNetwork {
    struct ActivationFunction {
        let function: (Matrix) -> Matrix
        let derivative: (Matrix) -> Matrix
    }

    struct Layer {
        var weights: Matrix
        var biases: Matrix

        let neuronCount: Int
        var activationFunction: ActivationFunction

        init(previousLayerSize: Int, neurons: Int, activationFunction: ActivationFunction) {
            // Initialize weights and biases randomly
            self.weights = Matrix.gaussianRandom(rows: neurons, columns: previousLayerSize)
            self.biases = Matrix.gaussianRandom(rows: neurons, columns: 1)

            self.activationFunction = activationFunction
            self.neuronCount = neurons
        }
    }

    private let inputLayerNeuronCount: Int
    private let outputLayerSize: Int

    private var layers: [Layer] = []

    init(inputLayerNeuronCount: Int, outputLayerSize: Int) {
        self.inputLayerNeuronCount = inputLayerNeuronCount
        self.outputLayerSize = outputLayerSize
    }

    mutating func addHiddenLayer(withNeuronCount neurons: Int, activationFunction: ActivationFunction) {
        layers.append(Layer(previousLayerSize: layers.last?.neuronCount ?? self.inputLayerNeuronCount, neurons: neurons, activationFunction: activationFunction))
    }

    /// Forward propagation
    func predictions(usingData data: [Double]) -> Matrix {
        precondition(!layers.isEmpty)

        let inputData = Matrix([data])
        return forwardPropagation(inputData: inputData).last!.activationFunctionApplied
    }

    /// Gradient descent
    /// - Parameters:
    ///   - trainingData: The data to feed ot the input layer of the NN (X). Each column must be an example, with its rows being the data.
    ///   - validationData: The expected data at the output of the NN (Y). Each row must be an example, with only 1 column.
    ///   - iterations: How many iterations to run.
    mutating func train(usingTrainingData trainingData: Matrix, validationData: Matrix, iterations: Int, alpha: Double) {
        precondition(!layers.isEmpty)
        assert(trainingData.columns == validationData.rows)

        for i in 0..<iterations {
            let forwardProp = forwardPropagation(inputData: trainingData)
            let backwardsProp = backwardsPropagation(usingTrainingData: trainingData, validationData: validationData, forwardPropagationResults: forwardProp)

            for (layerIndex, layerBackPropResult) in backwardsProp.enumerated() {
                self.updateParameters(in: &self.layers[layerIndex], using: layerBackPropResult, alpha: alpha)
            }

            if i.isMultiple(of: 10) {
                print("Iteration \(i)")

                getPredictions(usingOutput: forwardProp.last!.activationFunctionApplied)
            }
        }
    }
}

// MARK: - Implementation

private extension NeuralNetwork {
    struct LayerForwardPropagationResult {
        let weightsApplied: Matrix // Zn
        let activationFunctionApplied: Matrix // An
    }

    struct LayerBackwardPropagationResult {
        let layerMatrixDelta: Matrix // dZn
        let weightDelta: Matrix // dWn
        let biasDelta: Double // dbn
    }

    /// - Returns: A `LayerForwardPropagationResult` value for each of the layers that the data flows through.
    func forwardPropagation(inputData: Matrix) -> [LayerForwardPropagationResult] {
        var results: [LayerForwardPropagationResult] = []

        var nextLayerInput = inputData

        for layer in layers {
            let weightsApplied = layer.weights ° nextLayerInput + layer.biases
            let activations = layer.activationFunction.function(weightsApplied)
            nextLayerInput = activations

            results.append(.init(weightsApplied: weightsApplied, activationFunctionApplied: activations))
        }

        return results
    }

    func backwardsPropagation(usingTrainingData trainingData: Matrix, validationData: Matrix, forwardPropagationResults: [LayerForwardPropagationResult])  -> [LayerBackwardPropagationResult] {
        assert(forwardPropagationResults.count == layers.count)

        let m = Double(trainingData.rows)

        var backwardPropagationResults: [LayerBackwardPropagationResult] = []

        for (index, layer) in layers.enumerated().reversed() {
            let isFirstLayer = index == 0
            let isLastLayer = index == layers.count - 1
            let forwardPropagationData = forwardPropagationResults[index]

            let previousLayerActivationData = isFirstLayer ? trainingData : forwardPropagationResults[index - 1].activationFunctionApplied

            let layerMatrixDelta = isLastLayer
            ? forwardPropagationData.activationFunctionApplied - validationData.oneHot(withOutputLayerSize: outputLayerSize)
            : layers[index + 1].weights′ ° backwardPropagationResults.last!.layerMatrixDelta * layer.activationFunction.derivative(forwardPropagationResults[index].weightsApplied) // dZn
            let layerWeightDelta = 1 / m * layerMatrixDelta ° previousLayerActivationData′ // dWn
            let layerBiasDelta = 1 / m * layerMatrixDelta.sum()

            backwardPropagationResults.append(.init(layerMatrixDelta: layerMatrixDelta, weightDelta: layerWeightDelta, biasDelta: layerBiasDelta))
        }

        return backwardPropagationResults.reversed()
    }

    func updateParameters(in layer: inout Layer, using backwardPropagationResult: LayerBackwardPropagationResult, alpha: Double) {
        layer.weights -= alpha * backwardPropagationResult.weightDelta
        layer.biases -= alpha * backwardPropagationResult.biasDelta
    }

    func getPredictions(usingOutput output: Matrix) {
        print("output:")
        print("\(output)")
    }
}

extension NeuralNetwork.ActivationFunction {
    static let reLU = NeuralNetwork.ActivationFunction(
        function: { $0.ReLU() },
        derivative: {
            $0.map { $0 > 0 ? 1 : 0 }
        }
    )

    static let softMax = NeuralNetwork.ActivationFunction(
        function: {
            let inputExp = exp($0)
            return inputExp / inputExp.sum()
        },
        derivative: { _ in
            fatalError("TBD")
        }
    )
}

extension Matrix {
    private struct MatrixMirror {
        let rows: Int
        let columns: Int
        var values: [Double]

        var matrix: Matrix {
            return Matrix(rows: rows, columns: columns, values: values)
        }

        init(from matrix: Matrix) {
            self = unsafeBitCast(matrix, to: MatrixMirror.self)
        }
    }

    func ReLU() -> Matrix {
        // A matrix where every element is at least 0
        return maxel(0, self)
    }

    func sum() -> Double {
        return vDSP.sum(mutableValues)
    }

    var mutableValues: [Double] {
        get { return MatrixMirror(from: self).values }
        set {
            var mirror = MatrixMirror(from: self)
            mirror.values = newValue
            self = mirror.matrix
        }
    }

    func map(_ f: (Double) -> Double) -> Matrix {
        var copy = self
        copy.mutableValues = copy.mutableValues.map(f)
        return copy
    }
}

private extension Matrix {
    func oneHot(withOutputLayerSize outputLayerSize: Int) -> Matrix {
        assert(rows > 1)

        // Self is the validation data, with the following format: n rows, 1 column
        // [1, 7, 3, 9, 4, 1, 5...]
        //
        // The goal is to return a Matrix with 10 rows where each column is 1 if it's the right index for that sample

        var oneHot = Matrix(
            rows: outputLayerSize,
            columns: rows,
            repeatedValue: 0)

        for row in 0..<oneHot.rows {
            for col in 0..<oneHot.columns {
                if self[col, 0] == Double(row + 1) {
                    oneHot[row, col] = 1
                }

            }
        }

        assert(oneHot.rows == outputLayerSize)
        assert(oneHot.columns == self.rows)

        return oneHot
    }
}

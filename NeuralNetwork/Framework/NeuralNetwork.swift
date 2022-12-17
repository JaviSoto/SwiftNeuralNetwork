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
        var weights: Matrix {
            didSet {
                assert(weights.rows == neuronCount)
                assert(weights.columns == oldValue.columns)
            }
        }
        var biases: Matrix {
            didSet {
                assert(biases.rows == neuronCount)
                assert(biases.columns == oldValue.columns)
            }
        }

        let neuronCount: Int
        var activationFunction: ActivationFunction

        init(previousLayerSize: Int, neurons: Int, activationFunction: ActivationFunction) {
            // Initialize weights and biases randomly
            self.weights = Matrix.random(rows: neurons, columns: previousLayerSize) - 0.5
            self.biases = Matrix.random(rows: neurons, columns: 1) - 0.5

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

        let inputData = Matrix([data])′
        return forwardPropagation(inputData: inputData).last!.activationFunctionApplied
    }

    /// Gradient descent
    /// - Parameters:
    ///   - trainingData: The data to feed ot the input layer of the NN (X). Each column must be an example, with its rows being the data.
    ///   - validationData: The expected data at the output of the NN (Y). Each row must be an example, with only 1 column.
    ///   - iterations: How many iterations to run.
    ///   - learnignRate: How much change each iteration should have towards learning.
    ///   - progressObserver: An object you can observe to get updates on the training.
    mutating func train(usingTrainingData trainingData: Matrix, validationData: Matrix, iterations: Int, learningRate: Double, progressObserver: TrainingProgressObserver) {
        precondition(!layers.isEmpty)
        assert(trainingData.columns == validationData.rows)
        assert(validationData.rows > validationData.columns)
        assert(layers.last!.neuronCount == outputLayerSize)

        progressObserver.accuracies = []
        progressObserver.accuracies.reserveCapacity(iterations)

        for i in 0..<iterations {
            let forwardProp = forwardPropagation(inputData: trainingData)
            let backwardsProp = backwardsPropagation(usingTrainingData: trainingData, validationData: validationData, forwardPropagationResults: forwardProp)

            for (layerIndex, layerBackPropResult) in backwardsProp.enumerated() {
                self.updateParameters(inLayerAtIndex: layerIndex, using: layerBackPropResult, learningRate: learningRate)
            }

            let neuralNetworkOutput = forwardProp.last!.activationFunctionApplied
            let accuracy = accuracy(ofOutput: neuralNetworkOutput, againstValidationData: validationData)

            progressObserver.accuracies.append(accuracy)

            if i.isMultiple(of: 10) {
                print("Iteration \(i). Accuracy: \(Int(accuracy * 100))%")
            }
        }
    }

    func accuracy(usingInputData inputData: Matrix, expectedOutput: Matrix) -> Double {
        precondition(!layers.isEmpty)
        assert(inputData.columns == expectedOutput.rows)
        assert(expectedOutput.rows > expectedOutput.columns)

        let forwardProp = forwardPropagation(inputData: inputData)
        let neuralNetworkOutput = forwardProp.last!.activationFunctionApplied
        return accuracy(ofOutput: neuralNetworkOutput, againstValidationData: expectedOutput)
    }
}

// MARK: - Progress Reporting

extension NeuralNetwork {
    typealias TrainingSessionAccuracyProgress = [Double]

    final class TrainingProgressObserver: ObservableObject {
        @Published
        fileprivate(set) var accuracies: TrainingSessionAccuracyProgress = []

        init() {}
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
        assert(inputData.rows == inputLayerNeuronCount)

        var results: [LayerForwardPropagationResult] = []

        var nextLayerInput = inputData

        for layer in layers {
            let weightsApplied = layer.weights ° nextLayerInput + layer.biases // Zn
            let activations = layer.activationFunction.function(weightsApplied).assertValid() // An
            nextLayerInput = activations

            assert(weightsApplied.rows == layer.neuronCount)
            assert(activations.rows == layer.neuronCount)
            assert(weightsApplied.columns == inputData.columns)
            assert(activations.columns == inputData.columns)

            results.append(.init(weightsApplied: weightsApplied, activationFunctionApplied: activations))
        }

        return results
    }

    func backwardsPropagation(usingTrainingData trainingData: Matrix, validationData: Matrix, forwardPropagationResults: [LayerForwardPropagationResult])  -> [LayerBackwardPropagationResult] {
        assert(forwardPropagationResults.count == layers.count)

        let trainingDataPoints = Double(trainingData.columns) // m

        var backwardPropagationResults: [LayerBackwardPropagationResult] = []

        for (index, layer) in layers.enumerated().reversed() {
            let isFirstLayer = index == 0
            let isLastLayer = index == layers.count - 1
            let forwardPropagationData = forwardPropagationResults[index]

            let previousLayerActivationData = isFirstLayer ? trainingData : forwardPropagationResults[index - 1].activationFunctionApplied

            let layerMatrixDelta = isLastLayer
            ? forwardPropagationData.activationFunctionApplied - validationData.oneHot(withOutputLayerSize: outputLayerSize)
            : layers[index + 1].weights′ ° backwardPropagationResults.last!.layerMatrixDelta * layer.activationFunction.derivative(forwardPropagationResults[index].weightsApplied) // dZn
            let layerWeightDelta = (1 / trainingDataPoints * layerMatrixDelta).assertValid() ° previousLayerActivationData′
            // dWn
            let layerBiasDelta = 1 / trainingDataPoints * layerMatrixDelta.sum() // dbn

            assert(layerMatrixDelta.rows == layer.neuronCount)
            assert(layerMatrixDelta.columns == trainingData.columns)
            assert(layerMatrixDelta.rows == layer.weights.rows)

            layerBiasDelta.assertValid()

            assert(layerWeightDelta.rows == layer.neuronCount)
            assert(layerWeightDelta.rows == layer.weights.rows)
            assert(layerWeightDelta.columns == layer.weights.columns)

            backwardPropagationResults.append(.init(layerMatrixDelta: layerMatrixDelta, weightDelta: layerWeightDelta, biasDelta: layerBiasDelta))
        }

        return backwardPropagationResults.reversed()
    }

    mutating func updateParameters(inLayerAtIndex layerIndex: Int, using backwardPropagationResult: LayerBackwardPropagationResult, learningRate: Double) {
        layers[layerIndex].weights -= learningRate * backwardPropagationResult.weightDelta
        layers[layerIndex].biases -= learningRate * backwardPropagationResult.biasDelta
    }


    /// - Parameter output: The output of the neural network. The shape is [outputLayerSize rows, trainingData columns]
    /// - Returns:A `Matrix` with a row for each training data point whose value is the predicted value for it.
    func getPredictions(usingOutput output: Matrix) -> Matrix {
        var predictedValues: [[Double]] = []
        predictedValues.reserveCapacity(output.columns)

        for column in 0..<output.columns {
            var maxPredictedValue: Double = .leastNormalMagnitude
            var predictedValue: Double = 0

            for row in 0..<output.rows {
                let value = output[row, column]
                assert(value > 0)
                if value > maxPredictedValue {
                    maxPredictedValue = value
                    predictedValue = Double(row + 1)
                }
            }

            predictedValues.append([predictedValue])
        }

        let result = Matrix(predictedValues)

        assert(result.rows == output.columns)
        assert(result.columns == 1)

        return result
    }

    func accuracy(ofOutput output: Matrix, againstValidationData validationData: Matrix) -> Double {
        let predictions = getPredictions(usingOutput: output)
        assert(predictions.rows == validationData.rows)
        assert(predictions.columns == validationData.columns)
        assert(predictions.rows > 1)

        return Double(zip(validationData.mutableValues, predictions.mutableValues)
            .numberOfElements(matching: { $0 == $1 }))
        / Double(predictions.rows)
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
            let inputExp = exp($0 - max($0))
            // Add a small epsilon to the demominator to avoid division by 0
            let result = inputExp / (inputExp.sumMatrix() + 1e-5)
            return result.assertValid()
        },
        derivative: { _ in
            fatalError()
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
        let sum = vDSP.sum(mutableValues)
        return sum.isFinite ? sum : Double.greatestFiniteMagnitude
    }

    var average: Double {
        var average: Double = 0
        let values = mutableValues
        let count = values.count

        for value in values {
            average += value / Double(count)
        }

        return average
    }

    func sumMatrix() -> Matrix {
        var result = Matrix(rows: 1, columns: columns, repeatedValue: 0)

        for row in 0..<rows {
            for column in 0..<columns {
                result[0, column] += self[row, column]
            }
        }

        return result
    }

    var mutableValues: [Double] {
        get {
            return MatrixMirror(from: self).values
        }
        set {
            newValue.forEach { $0.assertValid() }

            var mirror = MatrixMirror(from: self)
            mirror.values = newValue
            self = mirror.matrix
        }
    }

    func map(_ f: (Double) -> Double) -> Matrix {
        var copy = self
        copy.mutableValues = copy.mutableValues.map { value in
            let newValue = f(value)
            newValue.assertValid()
            return newValue
        }
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

private extension Sequence {
    func numberOfElements(matching f: (Element) -> Bool) -> Int {
        var count = 0
        for element in self {
            if f(element) {
                count += 1
            }
        }

        return count
    }
}

extension Matrix {
    var shape: String {
        return "(\(rows), \(columns))"
    }
}

private extension Matrix {
    func assertValid(file: StaticString = #file, line: UInt = #line) -> Matrix {
        #if DEBUG
        mutableValues.forEach { $0.assertValid(file: file, line: line) }
        #endif
        return self
    }
}

private extension Double {
    func assertValid(file: StaticString = #file, line: UInt = #line) {
        assert(!self.isNaN, file: file, line: line)
        assert(!self.isInfinite, file: file, line: line)
    }
}

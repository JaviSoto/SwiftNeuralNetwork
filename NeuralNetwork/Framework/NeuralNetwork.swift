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
        let name: String
        let function: (Matrix) -> Matrix
        let derivative: (Matrix) -> Matrix
    }

    struct Layer {
        fileprivate(set) var weights: Matrix {
            didSet {
                precondition(weights.rows == neuronCount)
                precondition(weights.columns == oldValue.columns)
            }
        }
        fileprivate(set) var biases: Matrix {
            didSet {
                precondition(biases.rows == neuronCount)
                precondition(biases.columns == oldValue.columns)
            }
        }

        let neuronCount: Int
        fileprivate(set) var activationFunction: ActivationFunction

        init(previousLayerSize: Int, neurons: Int, activationFunction: ActivationFunction) {
            // Initialize weights and biases randomly
            self.weights = Matrix.random(rows: neurons, columns: previousLayerSize) - 0.5
            self.biases = Matrix.random(rows: neurons, columns: 1) - 0.5

            self.activationFunction = activationFunction
            self.neuronCount = neurons
        }
    }

    let inputLayerNeuronCount: Int
    let outputLayerSize: Int

    private(set) var layers: [Layer] = []

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
    ///   - limitToSamples: A number equal or less than the size of `trainingData` to use in each iteration.
    ///   - iterations: How many iterations to run.
    ///   - learnignRate: How much change each iteration should have towards learning.
    ///   - progressObserver: An object you can observe to get updates on the training.
    mutating func train(usingTrainingData trainingData: Matrix, validationData: Matrix, limitToSamples sampleLimit: Int, iterations: Int, learningRate: Double, progressObserver: TrainingProgressObserver) {
        print("Starting to train with training elements = \(trainingData.columns), batch size = \(sampleLimit), iterations = \(iterations), learning rate = \(learningRate)")

        precondition(!layers.isEmpty)
        precondition(trainingData.columns == validationData.rows)
        precondition(validationData.rows > validationData.columns)
        precondition(layers.last!.neuronCount == outputLayerSize)
        precondition(sampleLimit <= trainingData.columns)

        progressObserver.accuracies = []
        progressObserver.layerState = self.layers.map { .init(layer: $0, forwardPropagation: nil) }
        progressObserver.accuracies.reserveCapacity(iterations)
        progressObserver.shouldStopTraining.value = false

        for i in 0..<iterations {
            guard !progressObserver.shouldStopTraining.value else {
                print("Stopped training after \(i) iterations")
                break
            }

            let trainingIndicesToUse = (0..<validationData.rows).shuffled().prefix(sampleLimit)
            let inputData = trainingData.columns(trainingIndicesToUse)
            let validationData = validationData.rows(trainingIndicesToUse)

            precondition(trainingData.columns == validationData.rows)

            let forwardProp = forwardPropagation(inputData: inputData)
            let backwardsProp = backwardsPropagation(usingTrainingData: inputData, validationData: validationData, forwardPropagationResults: forwardProp)
 
            for (layerIndex, layerBackPropResult) in backwardsProp.enumerated() {
                self.updateParameters(inLayerAtIndex: layerIndex, using: layerBackPropResult, learningRate: learningRate)
            }

            let neuralNetworkOutput = forwardProp.last!.activationFunctionApplied
            let accuracy = Self.accuracy(ofOutput: neuralNetworkOutput, againstValidationData: validationData)

            progressObserver.accuracies.append(accuracy)
            progressObserver.layerState = zip(layers, forwardProp).map { .init(layer: $0, forwardPropagation: $1) }

            if i.isMultiple(of: 10) {
                print("Iteration \(i). Accuracy: \(Int(accuracy * 100))%")
            }
        }
    }

    func accuracy(usingInputData inputData: Matrix, expectedOutput: Matrix) -> Double {
        precondition(!layers.isEmpty)
        precondition(inputData.columns == expectedOutput.rows)
        precondition(expectedOutput.rows > expectedOutput.columns)

        let forwardProp = forwardPropagation(inputData: inputData)
        let neuralNetworkOutput = forwardProp.last!.activationFunctionApplied
        return Self.accuracy(ofOutput: neuralNetworkOutput, againstValidationData: expectedOutput)
    }
}

// MARK: - Progress Reporting

extension NeuralNetwork {
    typealias TrainingSessionAccuracyProgress = [Double]

    final class TrainingProgressObserver: ObservableObject {
        struct LayerState {
            let layer: Layer
            let forwardPropagation: NeuralNetwork.LayerForwardPropagationResult?
        }

        let shouldStopTraining: AtomicBool = false

        @Published
        fileprivate(set) var accuracies: TrainingSessionAccuracyProgress = []

        @Published
        fileprivate(set) var layerState: [LayerState] = []

        init() {}
    }
}

// MARK: - Implementation

extension NeuralNetwork {
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
    private func forwardPropagation(inputData: Matrix) -> [LayerForwardPropagationResult] {
        precondition(inputData.rows == inputLayerNeuronCount)

        var results: [LayerForwardPropagationResult] = []

        var nextLayerInput = inputData

        for layer in layers {
            let weightsApplied = (layer.weights ° nextLayerInput) + layer.biases // Zn
            let activations = layer.activationFunction.function(weightsApplied).assertValid() // An
            nextLayerInput = activations

            precondition(weightsApplied.rows == layer.neuronCount)
            precondition(activations.rows == layer.neuronCount)
            precondition(weightsApplied.columns == inputData.columns)
            precondition(activations.columns == inputData.columns)

            results.append(.init(weightsApplied: weightsApplied, activationFunctionApplied: activations))
        }

        return results
    }

    private func backwardsPropagation(usingTrainingData trainingData: Matrix, validationData: Matrix, forwardPropagationResults: [LayerForwardPropagationResult])  -> [LayerBackwardPropagationResult] {
        precondition(forwardPropagationResults.count == layers.count)

        let trainingDataPoints = Double(trainingData.columns) // m

        var backwardPropagationResults: [LayerBackwardPropagationResult] = []

        for (index, layer) in layers.enumerated().reversed() {
            let isFirstLayer = index == 0
            let isLastLayer = index == layers.count - 1
            let forwardPropagationData = forwardPropagationResults[index]

            let previousLayerActivationData = isFirstLayer ? trainingData : forwardPropagationResults[index - 1].activationFunctionApplied

            let layerMatrixDelta: Matrix  // dZn
            if !isLastLayer {
                layerMatrixDelta = ((layers[index + 1].weights′) ° backwardPropagationResults.last!.layerMatrixDelta) * layer.activationFunction.derivative(forwardPropagationResults[index].weightsApplied)
            } else {
                layerMatrixDelta = forwardPropagationData.activationFunctionApplied - validationData.oneHot(withOutputLayerSize: outputLayerSize)
            }
            let layerWeightDelta = (1 / trainingDataPoints) * (layerMatrixDelta ° (previousLayerActivationData′)) // dWn
            let layerBiasDelta = (1 / trainingDataPoints) * layerMatrixDelta.sum() // dbn

            precondition(layerMatrixDelta.rows == layer.neuronCount)
            precondition(layerMatrixDelta.columns == trainingData.columns)
            precondition(layerMatrixDelta.rows == layer.weights.rows)

            layerBiasDelta.assertValid()

            precondition(layerWeightDelta.rows == layer.neuronCount)
            precondition(layerWeightDelta.rows == layer.weights.rows)
            precondition(layerWeightDelta.columns == layer.weights.columns)

            backwardPropagationResults.append(.init(layerMatrixDelta: layerMatrixDelta, weightDelta: layerWeightDelta, biasDelta: layerBiasDelta))
        }

        return backwardPropagationResults.reversed()
    }

    private mutating func updateParameters(inLayerAtIndex layerIndex: Int, using backwardPropagationResult: LayerBackwardPropagationResult, learningRate: Double) {
        layers[layerIndex].weights -= learningRate * backwardPropagationResult.weightDelta
        layers[layerIndex].biases -= learningRate * backwardPropagationResult.biasDelta
    }


    /// - Parameter output: The output of the neural network. The shape is [outputLayerSize rows, trainingData columns]
    /// - Returns:A `Matrix` with a row for each training data point whose value is the predicted value for it.
    private static func getPredictions(usingOutput output: Matrix) -> Matrix {
        var predictedValues: [[Double]] = []
        predictedValues.reserveCapacity(output.columns)

        for column in 0..<output.columns {
            var maxPredictedValue: Double = .leastNormalMagnitude
            var predictedValue: Double = 0

            for row in 0..<output.rows {
                let value = output[row, column]
                precondition(value >= 0)
                if value > maxPredictedValue {
                    maxPredictedValue = value
                    predictedValue = Double(row)
                }
            }

            precondition(predictedValue >= 0)
            precondition(predictedValue <= 9)
            predictedValues.append([predictedValue])
        }

        let result = Matrix(predictedValues)

        precondition(result.rows == output.columns)
        precondition(result.columns == 1)

        return result
    }
}

extension NeuralNetwork {
    internal // @testable
    static func accuracy(ofOutput output: Matrix, againstValidationData validationData: Matrix) -> Double {
        precondition(validationData.rows == output.columns)
        precondition(validationData.columns == 1)
        precondition(validationData.rows > 1)
        precondition(output.columns == validationData.rows)

        let predictions = getPredictions(usingOutput: output)
        precondition(predictions.shape == validationData.shape)

        return Double(zip(validationData.mutableValues, predictions.mutableValues)
            .numberOfElements(matching: { $0 == $1 }))
        / Double(predictions.rows)
    }
}

extension NeuralNetwork.ActivationFunction {
    static let reLU = NeuralNetwork.ActivationFunction(
        name: "ReLU",
        function: { $0.ReLU() },
        derivative: {
            $0.map { $0 > 0 ? 1 : 0 }
        }
    )

    static let softMax = NeuralNetwork.ActivationFunction(
        name: "softMax",
        function: {
            let inputExp = exp($0)
            // Add a small epsilon to the demominator to avoid division by 0
            let result = inputExp / (inputExp.sumMatrix() + 1e-5)
            return result.assertValid()
        },
        derivative: { _ in
            fatalError()
        }
    )
}

internal // @testable
extension Matrix {
    func oneHot(withOutputLayerSize outputLayerSize: Int) -> Matrix {
        let numberOfSamples = rows
        precondition(numberOfSamples > 1)
        precondition(columns == 1)

        // Self is the validation data, with the following format: n rows, 1 column
        // [1, 7, 3, 9, 4, 1, 5...]
        //
        // The goal is to return a Matrix with 10 rows where each column is 1 if it's the right index for that sample

        var oneHot = Matrix(
            rows: outputLayerSize,
            columns: numberOfSamples,
            repeatedValue: 0)

        for sample in 0..<numberOfSamples {
            let sampleValue = Int(exactly: self[sample, 0])!
            oneHot[sampleValue, sample] = 1
        }

        precondition(oneHot.rows == outputLayerSize)
        precondition(oneHot.columns == self.rows)

        return oneHot
    }
}

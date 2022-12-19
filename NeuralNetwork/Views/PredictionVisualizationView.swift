//
//  PredictionVisualizationView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct PredictionVisualizationView: View {
    struct PredictionAttempt {
        let image: SampleImage
        let expectedLabel: Label?
    }

    let attempt: PredictionAttempt?

    @Binding
    var predictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome

    @Binding
    var tableOrder: [KeyPathComparator<ImageRecognitionNeuralNetwork.PredictionOutcome.Digit>]

    var body: some View {
        VStack {
            if let attempt {
                VStack {
                    VStack {
                        if let expectedLabel = attempt.expectedLabel {
                            let isCorrectPrediction = predictionOutcome.highestDigit.value == expectedLabel.representedNumber

                            Text("Data label: \(expectedLabel.representedNumber)")
                                .font(.title)

                            HStack {
                                PredictionCorrectnessView(predictionOutcome: predictionOutcome, expectedLabel: expectedLabel)
                                neuralNetOutputView
                                    .foregroundColor(isCorrectPrediction ? .green : .red)
                            }
                        } else {
                            neuralNetOutputView
                        }

                        PixelArrayImageView(sampleImage: attempt.image, lazy: false)
                            .frame(idealWidth: 300)
                    }

                    Table(predictionOutcome.digits, sortOrder: $tableOrder) {
                        TableColumn("Digit", value: \.value) { digit in
                            Text("\(digit.value)")
                                .bold(predictionOutcome.highestDigit.value == digit.value)
                        }
                        .width(50)

                        TableColumn("Confidence", value: \.confidence) { digit in
                            Text("\(digit.confidence.formatted(.percent.precision(.significantDigits(3))))")
                                .bold(predictionOutcome.highestDigit.value == digit.value)
                        }
                    }
                    .frame(minHeight: 350)
                    .onChange(of: tableOrder) { newOrder in
                        predictionOutcome.digits.sort(using: newOrder)
                    }
                }
            }

            Spacer()

            Divider()

        }
        .padding(.vertical)
    }

    private var neuralNetOutputView: some View {
        Text("Neural Net Output: \(predictionOutcome.highestDigit.value)")
            .font(.title)
    }
}

#if DEBUG

struct PredictionVisualizationView_Previews: PreviewProvider {
    static var previews: some View {
        PredictionVisualizationView(
            attempt: .init(image: MNISTParser.DataSet.randomItem.image, expectedLabel: MNISTParser.DataSet.randomItem.label),
            predictionOutcome: .constant(.init(digits: (0...9).map { digit in
                    .init(value: digit, confidence: Double.random(in: 0...1))
            })),
            tableOrder: .constant([.init(\.confidence, order: .reverse)])
        )
    }
}

#endif

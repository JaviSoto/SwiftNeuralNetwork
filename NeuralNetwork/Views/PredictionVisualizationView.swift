//
//  PredictionVisualizationView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

struct PredictionVisualizationView: View {
    let item: MNISTParser.DataSet.Item?
    let imageWidth: UInt32

    @Binding
    var predictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome

    @Binding
    var tableOrder: [KeyPathComparator<ImageRecognitionNeuralNetwork.PredictionOutcome.Digit>]

    var body: some View {
        VStack {
            if let item {
                VStack {
                    VStack {
                        let isCorrectPrediction = predictionOutcome.highestDigit.value == item.label.representedNumber

                        Text("Data label: \(item.label.representedNumber)")
                            .font(.title)

                        HStack {
                            PredictionCorrectnessView(predictionOutcome: predictionOutcome, expectedLabel: item.label)
                            Text("Neural Net Output: \(predictionOutcome.highestDigit.value)")
                                .foregroundColor(isCorrectPrediction ? .green : .red)
                                .font(.title)
                        }

                        SampleImageView(sampleImage: item.image, width: imageWidth, lazy: false)
                            .frame(width: 300)
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
                    .onChange(of: tableOrder) { newOrder in
                        predictionOutcome.digits.sort(using: newOrder)
                    }
                }
            }

            Spacer()

            Divider()

        }
        .navigationSubtitle(item.map { "Selected item '\(String(describing: $0.id))'" } ?? "")
        .padding(.vertical)
    }
}

#if DEBUG

struct PredictionVisualizationView_Previews: PreviewProvider {
    static var previews: some View {
        PredictionVisualizationView(
            item: MNISTParser.DataSet.randomItem,
            imageWidth: MNISTParser.DataSet.imageWidth,
            predictionOutcome: .constant(.init(digits: (0...9).map { digit in
                    .init(value: digit, confidence: Double.random(in: 0...1))
            })),
            tableOrder: .constant([.init(\.confidence, order: .reverse)])
        )
    }
}

#endif

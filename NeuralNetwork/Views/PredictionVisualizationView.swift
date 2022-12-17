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
    let updateImage: () -> Void

    @Binding
    var randomItemPredictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome

    @Binding
    var predictionOutcomeTableOrder: [KeyPathComparator<ImageRecognitionNeuralNetwork.PredictionOutcome.Digit>]

    var body: some View {
        VStack {
            if let item {
                VStack {
                    VStack {
                        Text("Data label: \(item.label.representedNumber)")
                            .font(.title)

                        SampleImageView(sampleImage: item.image, width: imageWidth)
                            .frame(width: 300)
                    }

                    Table(randomItemPredictionOutcome.digits, sortOrder: $predictionOutcomeTableOrder) {
                        TableColumn("Digit", value: \.value) { digit in
                            Text("\(digit.value)")
                                .bold(randomItemPredictionOutcome.highestDigit.value == digit.value)
                        }
                        .width(50)

                        TableColumn("Confidence", value: \.confidence) { digit in
                            Text("\(digit.confidence.formatted(.percent.precision(.significantDigits(3))))")
                                .bold(randomItemPredictionOutcome.highestDigit.value == digit.value)
                        }
                    }
                    .onChange(of: predictionOutcomeTableOrder) { newOrder in
                        randomItemPredictionOutcome.digits.sort(using: newOrder)
                    }
                    .frame(height: 300)
                }
            }

            Spacer()

            Divider()

            Button("Random image") {
                updateImage()
            }
        }
        .padding(.vertical)
    }
}

#if DEBUG

struct PredictionVisualizationView_Previews: PreviewProvider {
    static var previews: some View {
        PredictionVisualizationView(
            item: MNISTParser.DataSet.randomItem,
            imageWidth: MNISTParser.DataSet.imageWidth,
            updateImage: { },
            randomItemPredictionOutcome: .constant(.init(digits: (0...9).map { digit in
                    .init(value: digit, confidence: Double.random(in: 0...1))
            })),
            predictionOutcomeTableOrder: .constant([.init(\.confidence, order: .reverse)])
        )
    }
}

#endif

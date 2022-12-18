//
//  PredictionCorrectnessView.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI

enum PredictionCorrectness: Equatable, Comparable {
    case correct
    case incorrect

    init(predictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome, expectedLabel: Label) {
        let isCorrect = predictionOutcome.highestDigit.value == expectedLabel.representedNumber

        self = isCorrect ? .correct : .incorrect
    }
}

struct PredictionCorrectnessView: View {
    let correctness: PredictionCorrectness

    init(correctness: PredictionCorrectness) {
        self.correctness = correctness
    }

    init(predictionOutcome: ImageRecognitionNeuralNetwork.PredictionOutcome, expectedLabel: Label) {
        self.init(correctness: .init(predictionOutcome: predictionOutcome, expectedLabel: expectedLabel))
    }

    var body: some View {
        self.correctness.image
            .foregroundColor(self.correctness.color)
    }
}

private extension PredictionCorrectness {
    var image: Image {
        switch self {
        case .correct: return Image(systemName: "checkmark.circle.fill")
        case .incorrect: return Image(systemName: "x.circle.fill")
        }
    }

    var color: Color {
        switch self {
        case .correct: return .green
        case .incorrect: return .red
        }
    }
}

struct PredictionCorrectnessView_Previews: PreviewProvider {
    static var previews: some View {
        VStack {
            PredictionCorrectnessView(
                predictionOutcome: .init(digits: [
                    .init(value: 0, confidence: 0),
                    .init(value: 1, confidence: 1),
                ]),
                expectedLabel: .init(representedNumber: 1)
            )

            PredictionCorrectnessView(
                predictionOutcome: .init(digits: [
                    .init(value: 0, confidence: 0),
                    .init(value: 1, confidence: 1),
                ]),
                expectedLabel: .init(representedNumber: 2)
            )
        }
    }
}

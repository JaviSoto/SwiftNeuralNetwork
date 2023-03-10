//
//  LearningAccuracyEvolutionGraph.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import SwiftUI
import Charts

struct LearningAccuracyEvolutionGraph: View {
    let sessions: [NeuralNetwork.TrainingSessionAccuracyProgress]

    var body: some View {
        Chart(sessions.indexed) { session in
            ForEach(session.item.indexed) { accuracy in
                LineMark(
                    x: .value("Epoch", accuracy.index),
                    y: .value("Accuracy", Int(accuracy.item * 100))
                )
                .foregroundStyle(by: .value("Session", "\(session.index + 1)"))
            }
        }
        .chartYScale(domain: 0...100, type: .linear)
        .chartYAxis {
            AxisMarks(position: .leading, values: [0, 25, 50, 75, 100])
        }
        .chartXAxisLabel("Epoch", alignment: .bottom)
        .chartYAxisLabel("Accuracy %", alignment: .bottomLeading)
    }
}

struct LearningAccuracyEvolutionGraph_Previews: PreviewProvider {
    static var previews: some View {
        LearningAccuracyEvolutionGraph(sessions: [
            [0, 0.1, 0.4, 0.8, 0.9],
            [0, 0.2, 0.3, 0.4, 0.5],
            [0, 0.3, 0.6, 0.7, 0.8]
        ])
        .padding()
    }
}

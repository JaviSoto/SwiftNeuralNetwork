//
//  SwiftUIExtensions.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/15/22.
//

import SwiftUI

struct ValueSlider: View {
    var name: String
    @Binding
    var value: Double
    var range: ClosedRange<Double>
    var step: Double
    var decimalPoints = 1
    var unit: String = ""

    var body: some View {
        Slider(value: $value, in: range, step: step) {
            Text("\(name): \(value, specifier: "%.\(decimalPoints)f")\(unit)")
                .lineLimit(1)
        }
    }
}

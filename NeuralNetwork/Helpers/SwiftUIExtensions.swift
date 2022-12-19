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
                .minimumScaleFactor(0.8)
                .layoutPriority(10)
        }
    }
}

extension Color {
    static var background: Color {
        #if os(macOS)
        return Color(nsColor: NSColor.windowBackgroundColor)
        #elseif os(iOS)
        return Color(uiColor: UIColor.background)
        #endif
    }
}

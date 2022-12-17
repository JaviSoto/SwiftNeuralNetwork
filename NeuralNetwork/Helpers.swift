//
//  Helpers.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/15/22.
//

import Foundation

extension Int {
    var double: Double {
        get { return Double(self) }
        set { self = Int(exactly: newValue)! }
    }
}

func measure<T>(_ name: String, _ f: () -> T) -> T {
    print("Starting '\(name)'")

    let start = CFAbsoluteTimeGetCurrent()
    let result = f()
    let end = CFAbsoluteTimeGetCurrent()

    let diff = end - start
    let duration = diff < 1 ? "\(Int(diff * 1000))ms" : "\(diff)s"

    print("'\(name)' took \(duration)")

    return result
}

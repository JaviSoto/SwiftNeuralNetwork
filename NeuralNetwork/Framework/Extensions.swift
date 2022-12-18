//
//  Extensions.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/17/22.
//

import Foundation

extension Sequence {
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

extension Double {
    func assertValid(file: StaticString = #file, line: UInt = #line) {
        assert(!self.isNaN, file: file, line: line)
        assert(!self.isInfinite, file: file, line: line)
    }
}

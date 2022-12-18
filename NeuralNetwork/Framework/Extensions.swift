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

struct IndexedCollection<Base: RandomAccessCollection>: RandomAccessCollection where Base.Index: Hashable {
    typealias Index = Base.Index

    struct IndexedElement: Identifiable {
        let index: Base.Index
        let item: Base.Element

        var id: Base.Index {
            return index
        }
    }

    let base: Base

    init(_ base: Base) {
        self.base = base
    }

    var startIndex: Index { base.startIndex }

    // corrected typo: base.endIndex, instead of base.startIndex
    var endIndex: Index { base.endIndex }

    func index(after i: Index) -> Index {
        base.index(after: i)
    }

    func index(before i: Index) -> Index {
        base.index(before: i)
    }

    func index(_ i: Index, offsetBy distance: Int) -> Index {
        base.index(i, offsetBy: distance)
    }

    subscript(position: Index) -> IndexedElement {
        .init(index: position, item: base[position])
    }
}

extension RandomAccessCollection where Self.Index: Hashable {
    var indexed: IndexedCollection<Self> {
        return IndexedCollection(self)
    }
}

public extension Comparable {
    func clipped(minimum: Self, maximum: Self) -> Self {
        return self
            .clipped(minimum: minimum)
            .clipped(maximum: maximum)
    }

    func clipped(minimum: Self) -> Self {
        return max(self, minimum)
    }

    func clipped(maximum: Self) -> Self {
        return min(self, maximum)
    }
}

public extension ClosedRange where Bound == Double {
    func interpolation(for value: Double) -> Double {
        return Double(value - lowerBound) / Double(upperBound - lowerBound)
    }

    func interpolating(by value: Double) -> Double {
        return lowerBound + Bound(value) * (upperBound - lowerBound)
    }

    func clampedInterpolation(for value: Double) -> CGFloat {
        return interpolation(for: value.clipped(minimum: lowerBound, maximum: upperBound))
    }

    func projecting(clamped value: Double, into range: ClosedRange<Double>) -> Double {
        return range.interpolating(by: clampedInterpolation(for: value))
    }
}

import os

final class AtomicBool {
    private let lockedBool: OSAllocatedUnfairLock<Bool>

    init(_ value: Bool) {
        self.lockedBool = .init(initialState: value)
    }

    var value: Bool {
        get {
            return lockedBool.withLock { $0 }
        }
        set {
            lockedBool.withLock { $0 = newValue }
        }
    }
}

extension AtomicBool: ExpressibleByBooleanLiteral {
    convenience init(booleanLiteral value: BooleanLiteralType) {
        self.init(value)
    }
}

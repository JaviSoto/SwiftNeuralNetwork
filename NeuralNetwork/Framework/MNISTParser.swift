//
//  MNISTParser.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation

struct SampleImage {
    typealias Pixel = UInt8
    let pixels: [Pixel]
}

struct Label {
    let representedNumber: UInt8
}

struct MINSTFileHeader {
    let magicNumber: UInt32
    let numberOfItems: UInt32
}

struct TrainingImageSetFile {
    let header: MINSTFileHeader
    let numberOfRows: UInt32
    let numberOfColumns: UInt32
    let images: [SampleImage]
}

struct TrainingLabelsFile {
    let header: MINSTFileHeader
    let labels: [Label]
}

enum MNISTParser {
    static func parseImageSet(from url: URL, maxCount: Int?) throws -> TrainingImageSetFile {
        let data = try Data(contentsOf: url)
        var parser = BigEndianDataParser(data: data)

        let header = parser.parseFileHeader()

        let rows = parser.parseUInt32()
        let columns = parser.parseUInt32()

        let images = Array((0..<min(Int(header.numberOfItems), maxCount ?? .max)).lazy
            .map { _ in parser.parseArrayOfBytes(withCount: rows * columns) }
            .map(SampleImage.init))

        return TrainingImageSetFile(
            header: header,
            numberOfRows: rows,
            numberOfColumns: columns,
            images: images
        )
    }

    static func parseLabels(from url: URL, maxCount: Int?) throws -> TrainingLabelsFile {
        let data = try Data(contentsOf: url)
        var parser = BigEndianDataParser(data: data)

        let header = parser.parseFileHeader()

        let labels = parser.parseArrayOfBytes(withCount: min(header.numberOfItems, maxCount.map(UInt32.init) ?? .max))
            .map(Label.init)

        return TrainingLabelsFile(
            header: header,
            labels: labels
        )
    }

    struct DataSet {
        typealias Item = (image: SampleImage, label: Label)

        let imageWidth: UInt32
        var items: [Item]

        func shuffle() -> DataSet {
            var copy = self
            copy.items.shuffle()
            return copy
        }

        func cropped(maxLength: Int) -> DataSet {
            var copy = self
            copy.items = Array(items.prefix(maxLength))
            return copy
        }

        static func + (lhs: DataSet, rhs: DataSet) -> DataSet {
            assert(lhs.imageWidth == rhs.imageWidth)

            var items = lhs.items
            items.append(contentsOf: rhs.items)

            return DataSet(imageWidth: lhs.imageWidth, items: items)
        }
    }

    static func loadData(imageSetFileURL: URL, labelDataFileURL: URL, maxCount: Int?) throws -> DataSet {
        let imageSet = try parseImageSet(from: imageSetFileURL, maxCount: maxCount)
        let labels = try parseLabels(from: labelDataFileURL, maxCount: maxCount)

        assert(imageSet.images.count == labels.labels.count)

        let items = zip(imageSet.images, labels.labels).map { ($0, $1) }

        return DataSet(imageWidth: imageSet.numberOfRows, items: items)
    }
}

private struct BigEndianDataParser {
    private let data: Data
    private var currentIndex = 0

    init(data: Data) {
        self.data = data
    }

    mutating func parseUInt32() -> UInt32 {
        let value: UInt32 = parse()
        return value.byteSwapped
    }

    mutating func parseUInt8() -> UInt8 {
        return (parse() as UInt8).byteSwapped
    }

    mutating func parseFileHeader() -> MINSTFileHeader {
        let magicNumber = parseUInt32()
        let numberOfItems = parseUInt32()

        return MINSTFileHeader(magicNumber: magicNumber, numberOfItems: numberOfItems)
    }

    mutating func parseArrayOfBytes(withCount count: UInt32) -> [UInt8] {
        return (0..<count).map { _ in parseUInt8() }
    }

    private mutating func parse<T>() -> T {
        let size = MemoryLayout<T>.size
        let data = self.data[currentIndex..<(currentIndex + size)]
        currentIndex += size

        return data.withUnsafeBytes { buffer in
            return buffer.load(as: T.self)
        }
    }
}

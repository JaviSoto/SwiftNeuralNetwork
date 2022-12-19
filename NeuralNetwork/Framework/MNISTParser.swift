//
//  MNISTParser.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation

struct SampleImage: Equatable {
    typealias Pixel = UInt8
    let pixels: [Pixel]
    let width: UInt32
    let height: UInt32

    init(pixels: [Pixel], width: UInt32, height: UInt32) {
        assert(pixels.count == width * height)

        self.pixels = pixels
        self.width = width
        self.height = height
    }
}

struct Label: Equatable, Comparable {
    let representedNumber: UInt8

    static func < (lhs: Label, rhs: Label) -> Bool {
        return lhs.representedNumber < rhs.representedNumber
    }
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
            .map { SampleImage(pixels: $0, width: rows, height: columns) })

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
        enum Category: Hashable, Comparable, CaseIterable, CustomStringConvertible {
            case training
            case testing

            var description: String {
                switch self {
                case .training: return "Training"
                case .testing: return "Testing"
                }
            }
        }

        struct Item: Identifiable, Hashable {
            struct Identifier: Hashable, Comparable, CustomStringConvertible {
                let category: Category
                let value: Int

                var description: String {
                    return "\(category) \(value)"
                }

                static func < (lhs: Identifier, rhs: Identifier) -> Bool {
                    guard lhs.category == rhs.category else { return lhs.category < rhs.category }

                    return lhs.value < rhs.value
                }
            }

            let id: Identifier
            let image: SampleImage
            let label: Label

            init(id: Identifier, image: SampleImage, label: Label) {
                self.id = id
                self.image = image
                self.label = label
            }

            static func == (lhs: Item, rhs: Item) -> Bool {
                return lhs.id == rhs.id
            }

            func hash(into hasher: inout Hasher) {
                hasher.combine(id)
            }
        }

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
            precondition(lhs.imageWidth == rhs.imageWidth)

            var items = lhs.items
            items.append(contentsOf: rhs.items)

            return DataSet(imageWidth: lhs.imageWidth, items: items)
        }
    }

    static func loadData(imageSetFileURL: URL, labelDataFileURL: URL, category: DataSet.Category, maxCount: Int?) throws -> DataSet {
        let imageSet = try parseImageSet(from: imageSetFileURL, maxCount: maxCount)
        let labels = try parseLabels(from: labelDataFileURL, maxCount: maxCount)

        precondition(imageSet.images.count == labels.labels.count)

        let items = zip(imageSet.images, labels.labels).enumerated().map { (index, element) in
            DataSet.Item(id: .init(category: category, value: index + 1), image: element.0, label: element.1)
        }

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

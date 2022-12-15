//
//  ImageRendering.swift
//  NeuralNetwork
//
//  Created by Javier Soto on 12/14/22.
//

import Foundation
import SwiftUI

extension SampleImage {
    public func asSwiftUIImage(width: UInt32) -> SwiftUI.Image {
        let width = Int(width)
        let pixels = pixels.map(\.sRGBAValue)
        assert(pixels.count.isMultiple(of: width))

        let providerRef = CGDataProvider(data: NSData(bytes: pixels, length: pixels.count * 4))

        let cgImage = CGImage(
            width: width,
            height: pixels.count / width,
            bitsPerComponent: 8,
            bitsPerPixel: 4 * 8,
            bytesPerRow: 4 * width,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: .byteOrder32Little,
            provider: providerRef!,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!

        return Image(decorative: cgImage, scale: 1)
    }
}

extension SampleImage.Pixel {
    var sRGBAValue: UInt32 {
        let color = UInt32(self)
        return color << 24 + color << 16 + color << 8 + 255
    }
}

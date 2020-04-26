//
//  PolynomialRegressionModel.swift
//  Demo
//
//  Created by Cyril Garcia on 4/26/20.
//  Copyright © 2020 Cyril Garcia. All rights reserved.
//

import Foundation
import TensorFlow

struct PolynomialRegression: Layer {
    
    var a = Tensor([Float.random(in: 0...1)])
    var b = Tensor([Float.random(in: 0...1)])
    var c = Tensor([Float.random(in: 0...1)])
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
//        y = ax^2 + bx + c
        return (a * (input * input)) + (b * input) + c
    }
    
    mutating func clear() {
        a = Tensor([Float.random(in: 0...1)])
        b = Tensor([Float.random(in: 0...1)])
        c = Tensor([Float.random(in: 0...1)])
    }
}

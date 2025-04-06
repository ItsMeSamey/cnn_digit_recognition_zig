const std = @import("std");
const CNN = @import("cnn.zig").CNN;
const Layer = @import("layers.zig");
const Function = @import("functions.zig");
const MNIST = @import("read_minst.zig");

const cnn = CNN(f32, 28, 28, [_]Layer.LayerType{
  Layer.getFlattener(),
  Layer.getDense(32, Function.ReLU),
  Layer.getDense(16, Function.Sigmoid),
  Layer.getDense(10, Function.Softmax),
});

const MNISTIterator = MNIST.GetMinstIterator(28, 28);

pub fn main() !void {
  var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
  defer if (gpa.deinit() != .ok) @panic("Memory leak detected!!");
  const allocator = gpa.allocator();

  var trainer = cnn.Trainer{.layers = undefined};
  const mnist_iterator = try MNISTIterator.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_iterator.free(allocator);

  trainer.train(mnist_iterator, .{
    .verbose = true,
    .batch_size = 32,
    .learning_rate = 0.01,
  });
}


const std = @import("std");
const CNN = @import("cnn.zig").CNN;
const Layer = @import("layers.zig");
const Activation = @import("functions_activate.zig");
const Loss = @import("functions_loss.zig");
const MNIST = @import("read_minst.zig");

const cnn = CNN(f64, 28, 28, Loss.MeanSquaredError, [_]Layer.LayerType{
  Layer.getFlattener(),
  Layer.getDense(64, Activation.Tanh),
  Layer.getDense(32, Activation.Tanh),
  Layer.getDense(16, Activation.Sigmoid),
  Layer.getDense(10, Activation.Normalize),
});

const MNISTIterator = MNIST.GetMinstIterator(28, 28);

pub fn main() !void {
  @import("logger.zig").init();

  var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
  defer if (gpa.deinit() != .ok) @panic("Memory leak detected!!");
  const allocator = gpa.allocator();
  var rng = std.Random.DefaultPrng.init(0);

  var trainer = cnn.Trainer{.layers = undefined};
  const mnist_iterator = try MNISTIterator.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_iterator.free(allocator);

  trainer.train(mnist_iterator, .{
    .verbose = true,
    .batch_size = 32,
    .learning_rate = 1,
    .rng = rng.random(),
  });
}


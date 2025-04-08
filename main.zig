const std = @import("std");
const CNN = @import("cnn.zig").CNN;
const Layer = @import("layers.zig");
const Activation = @import("functions_activate.zig");
const Loss = @import("functions_loss.zig");
const MNIST = @import("read_minst.zig");

const cnn = CNN(f64, 28, 28, Loss.MeanSquaredError, [_]Layer.LayerType{
  Layer.getFlattener(),
  Layer.getDense(256, Activation.ReLU),
  Layer.getDense(128, Activation.Tanh),
  Layer.getDense(64, Activation.ReLU),
  Layer.getDense(16, Activation.Sigmoid),
  Layer.getDense(10, Activation.NormalizeSquared),
});

const MNISTIterator = MNIST.GetMinstIterator(28, 28);

pub fn main() !void {
  @import("logger.zig").init();

  var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
  defer if (gpa.deinit() != .ok) @panic("Memory leak detected!!");
  const allocator = gpa.allocator();
  var rng = std.Random.DefaultPrng.init(@bitCast(std.time.timestamp()));

  var trainer = cnn.Trainer{.layers = undefined};
  const mnist_iterator = try MNISTIterator.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_iterator.free(allocator);

  for (0..32) |i| {
    trainer.train(mnist_iterator, .{
      .verbose = true,
      .batch_size = @intCast(i + 8),
      .learning_rate = @as(f64, 1) / @as(f64, @floatFromInt(1 + i)),
      .rng = rng.random(),
    });
  }

  var tester = trainer.toTester();

  const mnist_test_iterator = try MNISTIterator.init("./datasets/t10k-images.idx3-ubyte", "./datasets/t10k-labels.idx1-ubyte", allocator);
  defer mnist_test_iterator.free(allocator);

  const loss = tester.@"test"(mnist_test_iterator, false);
  std.debug.print("\n>>Final Loss: {d:.3}\n", .{loss*100});
}


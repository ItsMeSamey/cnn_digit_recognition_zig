const std = @import("std");

pub fn CategoricalCrossentropy(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: u8) F {
      return -@log(predictions[target]);
    }

    pub fn backward(predictions: *const [LEN]F, target: u8, output: *[LEN]F) void {
      std.debug.assert(target < LEN);
      inline for (0..LEN) |i| {
        output[i] = if (i != target) 0 else -1/predictions[i];
      }
    }
  };
}

pub fn MeanSquaredError(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: u8) F {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        sum += if (i == target) (predictions[i]-1) * (predictions[i]-1) else predictions[i]*predictions[i];
      }
      return sum / LEN;
    }

    pub fn backward(predictions: *const [LEN]F, target: u8, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (i == target) 2 * (predictions[i] - 1) else 2 * predictions[i];
      }
    }
  };
}

pub fn MeanAbsoluteError(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: u8) F {
      var sum: F = 0;
      inline for (0..LEN) |i| {
        sum += @abs(if (i == target) predictions[i] - 1 else predictions[i]);
      }
      return sum / LEN;
    }

    pub fn backward(predictions: *const [LEN]F, target: u8, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        output[i] = if (predictions[i] > if (i == target) 1 else 0) 1 else -1;
      }
    }
  };
}

pub fn ClosestError(LEN: comptime_int, F: type) type {
  return struct {
    pub fn forward(predictions: *const [LEN]F, target: u8) F {
      var retval: F = 0;
      inline for (0..LEN) |i| {
        if (i == target) {
          if (predictions[i] < 0) {
            retval += -2 * predictions[i];
          } else if (predictions[i] < 1) {
            retval += 1 - predictions[i];
          } else {
            retval += (predictions[i] - 1) / 2;
          }
        } else {
          if (predictions[i] < 0) {
            retval += -predictions[i] / 2;
          } else {
            retval += @exp(predictions[i]);
          }
        }
      }
      return retval;
    }

    pub fn backward(predictions: *const [LEN]F, target: u8, output: *[LEN]F) void {
      inline for (0..LEN) |i| {
        if (i == target) {
          if (predictions[i] < 0) {
            output[i] = -2;
          } else if (predictions[i] < 1) {
            output[i] = -1;
          } else {
            output[i] = 0.5;
          }
        } else {
          if (predictions[i] < 0) {
            output[i] = -0.5;
          } else {
            output[i] = @exp(predictions[i]);
          }
        }
      }
    }
  };
}


class APIResponse {
    constructor(data, message) {
      this.data = data
      this.message = message
    }
    toJson() {
        return {
          data: this.data,
          message: this.message,
        };
      }
  }

  module.exports = APIResponse
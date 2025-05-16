package main
import (
    "fmt"
    "time"
)

func main() {
    done := make(chan bool)          // channel signals when work is done

    go func() {                      // **go** keyword launches a goroutine
        time.Sleep(time.Second)
        fmt.Println("token 1")
        done <- true                 // send signal
    }()

    <-done                            // main blocks here until signal arrives
}
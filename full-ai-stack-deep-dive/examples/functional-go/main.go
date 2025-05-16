package main

import (
    "flag" // standard-lib command-line flags
    "fmt"
    "os"
)

// greet returns the string we want to print.
// Keeping it as a function makes unit-testing painless.
func greet(who string) string {
    if who == "" {
        who = "world"
    }
    return fmt.Sprintf("Hello, %s!", who)
}

func main() {
    // ① define a string flag called -name
    nameFlag := flag.String("name", "", "name to greet")
    flag.Parse() // ② parse os.Args and populate nameFlag

    // ③ do the work
    msg := greet(*nameFlag)
    fmt.Println(msg)

    // ④ exit 0 on success (explicit for clarity)
    os.Exit(0)
}